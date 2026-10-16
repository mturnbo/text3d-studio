# AGENTS.md

Guidance for AI coding agents (and humans) working on **Text3D Studio**: a PyQt6 + OpenGL 4.1 core
desktop app that composites extruded 3D text into a photo with the *layer-stack duplication*
technique, then exports at the photo's native resolution. See `README.md` for the user-facing
overview and roadmap.

## Quick commands

```bash
uv sync                                  # install deps into .venv (never use pip)
uv run text3d-studio [image]             # launch the app; must run from the repo root
uv run python -m text3d_studio [image]   # same, without the console script
uv run pytest -q                         # full suite (~2–5 s), includes real-GPU tests
uv run pytest -q tests/test_render.py    # GPU/render tests only
```

- `uv run` only finds the console script from inside the project directory; elsewhere use
  `uv run --project /path/to/text3d-studio text3d-studio`.
- Python ≥ 3.11. Dependencies: `PyQt6`, `PyOpenGL`, `numpy`; dev: `pytest`. Do not add `scipy`,
  `Pillow`, `moderngl`, etc. without a clear need. Blurs are implemented in numpy, and image I/O goes through `QImage`.

## Architecture (MVVM)

```
model.py            Pure dataclasses: Scene{text, transform, ground, grid, material, light, shadow, camera}, Document.
                    No Qt, no OpenGL. Document.native_size is the export size.
viewmodel.py        SceneViewModel(QObject): the ONLY mutation point.
                      set(section, field, value) / nudge(...) / set_background(pixels, path)
                      signals: scene_changed(section, field), text_changed(), background_changed()
views/              Qt widgets only. Never touch GL state or numpy image math here.
  main_window.py      menus, status bar, file dialogs
  control_panel.py    declarative slider/colour/checkbox layout, two-way bound to the ViewModel
  image_canvas.py     QOpenGLWidget: owns the GL context, mouse→ViewModel edits, delegates drawing.
                      Holds the active tool (TEXT_TOOL / GRID_TOOL), which is editor state, not scene state
render/             OpenGL, no widget imports. Usable from any current GL context.
  gl_utils.py         ShaderProgram, UnitQuad, TriangleBuffer, Texture2D, ColorTarget, RenderTarget
  shaders.py          all GLSL 4.10 sources as strings
  text3d_renderer.py  instanced layer stack (beauty) + shadow casters (caps + side walls)
  shadow_renderer.py  half-res shadow mask → separable Gaussian blur → multiply composite
  compositor.py       frame orchestration: background → shadow → text
  exporter.py         offscreen 8× MSAA render at native size → resolve → glReadPixels
  grid_renderer.py    editor-only overlay: ground grid, text footprint and lanes, horizon,
                      vanishing-point rings. Drawn by ImageCanvas after Compositor.render
math3d.py           row-major numpy 4x4 helpers; to_gl() transposes to column-major float32.
                    Also image-space perspective: project_to_image / image_ray (pixels, origin
                    top-left, y down), intersect_ray_plane, 4-point homography, vanishing points,
                    focal_from_vanishing_points, ground_pose_from_rectangle, ground_angles_from_rotation
ground.py           Ground model B: ground pose (GroundParams) → plane, origin, light frame; text
                    placement on it (text_position, text_model_matrix, world_to_ground, text_distance)
lighting.py         LightingEngine: (light, ground) → world light dir, colours, ground plane, shadow matrix
text_raster.py      QFont → mask (RG edge dir, B bevel, A coverage) + traced side-wall triangles
image_io.py         QImage ⇄ numpy RGBA (top row first); SUPPORTED_EXTENSIONS for dialogs and drops
preferences.py      QSettings-backed last folder + recent files (max 8). MainWindow requires one.
app.py              configure_opengl() + main()
```

Data flow: widget → `vm.set()` → `scene_changed` → `canvas.update()` → `paintGL()` →
`Compositor.render(scene, fbo, viewport)`. The preview and the export call **the same**
`Compositor.render`, which is why they match. Keep it that way.

### Rendering pipeline

1. **Background**: full-screen quad. The viewport is letterboxed to the photo's aspect ratio, and the
   projection aspect always equals the photo aspect.
2. **Shadow** (if enabled): the casters are flattened onto the ground plane with
   `math3d.planar_shadow_matrix`, rendered at half resolution into an R8 target with `GL_MAX`
   blending (strength fades with height above the ground, controlled by `shadow.falloff`). The
   result is blurred twice in each direction, then **multiplied** onto the background.
3. **Text**: the glyph quad is drawn as one instanced call, with `layers` instances. Instance 0 is the
   beveled face, the middle instances are side walls (normal = outward edge direction, darkening
   with depth) and the last instance is the back cap. Depth test is on, with alpha-to-coverage.

### Coordinate conventions

- The camera is fixed at the origin looking down −Z, so world space = view space. Vertical FOV comes from `camera.fov`.
- Text local space is in em units, scaled by `transform.size`. The ink's bottom sits at local y=0 and is centered on x.
  The front face is at z=0 and the extrusion goes to `z = -depth` (depth is a *fraction of size*).
- **Ground model B**: the ground is posed relative to the camera by `scene.ground`
  (pitch, roll, yaw, height). Its origin is the point `height` below the camera, along the ground's up axis.
  The text stands on the ground at ground coordinates `(transform.x, transform.z)`. Negative z is away
  from the camera. `transform.rot_y` spins the text around the ground's up axis.
- Ground rotation = `Rz(roll) · Rx(pitch) · Ry(yaw)`. Text model matrix =
  `T(text_position) · ground rotation · Ry(transform.rot_y) · S(size)`. Always use `ground.py`;
  don't rebuild these by hand.
- The light is defined in the ground's **tilt frame** (`ground.light_frame`, pitch and roll only):
  azimuth 0 = toward the camera, elevation measured from the ground. Grid yaw and text yaw don't move
  the light, but tilting the ground does.
- **Ground edits keep the text where it appears.** `SceneViewModel.set("ground", …)` calls
  `ground.reanchor_text`. The text's anchor stays on the same camera ray, `size` scales with its distance
  along that ray (so apparent size is kept), and text yaw offsets grid yaw. Expect `transform.x/z/rot_y/size`
  change events after any ground edit. If the ray misses the new ground, the text is left as is.
- The default scene (pitch 0, height 1.5, text at z −12, yaw −20) renders pixel-identically to
  v0.5.0's defaults. `tests/test_ground.py` holds the legacy conversion if old values ever need migrating.
- Image arrays are `(H, W, 4) uint8`, top row first. Texture row 0 = v 0, so quads sample `1 - y`.
  Exports flip rows after `glReadPixels`.

## Hard-won gotchas (read before touching render code)

- **macOS caps OpenGL at 4.1 core.** Keep shaders at `#version 410 core`. Every draw needs a bound
  VAO. `configure_opengl()` must run **before** `QApplication` is created.
- **QOpenGLWidget renders into its own FBO.** Always rebind `self.defaultFramebufferObject()` after
  using offscreen targets. Viewport sizes must be multiplied by `devicePixelRatioF()` (HiDPI).
- **Mipmaps and alpha cutoff.** Strongly foreshortened quads pick high mip levels whose averaged alpha
  drops below 0.5, so every fragment is discarded. The shadow pass samples `textureLod(..., 0.0)`,
  and the beauty pass caps the LOD at 1.5. Don't switch these back to plain `texture()`.
- **Layer slices cast no shadow when the light is parallel to them.** Each slice projects to a
  sub-pixel sliver. That's why the shadow pass also draws side-wall quads traced from the glyph
  outline (`text_raster.extract_walls`). Removing them makes
  `test_grazing_light_still_casts_shadow_via_side_walls` fail. That's intentional.
- **Visible banding between slices.** `Text3DRenderer.effective_layers` raises the layer count to
  about 1.5 slices per on-screen pixel of extrusion (capped at 2048). The user's `layers` value is a
  minimum, not an exact count.
- **The planar shadow blows up at grazing elevations.** `LightingEngine` clamps the shadow light to
  `MIN_SHADOW_ELEVATION` (3°).
- **Keep samplers complete.** Upload a 1×1 placeholder before real data. An incomplete bound texture
  triggers macOS "unloadable texture" warnings.
- **Rasterizing is CPU work (~0.2 s).** Only `text.{content, font_family, bold, italic, bevel_width}`
  (`viewmodel.RASTER_FIELDS`) re-rasterize, and the canvas debounces them (`TEXT_DEBOUNCE_MS`).
  Everything else is a uniform and must stay cheap. Never re-rasterize per frame.
- **PyQt signal arguments.** `QAction.triggered` and `QAbstractButton.clicked` pass a `checked`
  bool. Don't connect them directly to methods with optional parameters; wrap them in a lambda.
- `ShaderProgram` caches uniform locations. Setting a uniform the shader optimized away is a
  harmless no-op (location −1).

### Editor overlays (the perspective grid)

- Overlays are **preview only**. They're drawn in `ImageCanvas.paintGL` *after* `Compositor.render` and
  must never be added to `Compositor.render`, or they'd appear in exports.
  `test_export_is_identical_with_grid_shown_or_hidden` guards this.
- They reuse the composite's depth buffer: depth test on, depth writes off, so the text hides them.
  The horizon pass sits at the far plane (z = 0.999999) for the same reason.
- Procedural lines use `fwidth` for constant pixel width (scaled by the device pixel ratio). With
  `fract()`-wrapped coordinates, take the derivative of the *unwrapped* value, or a false line shows
  up at every cell's half-way point.
- Grid cells fade out below ~4 px so they don't shimmer toward the horizon. The accent lanes and the
  horizon don't fade.

### Tools and canvas gestures

| Tool | Left-drag | Right-drag | Wheel |
|---|---|---|---|
| **Move Text** (V) | slide text on the ground (grabbed point stays under the cursor) | text yaw | size; Ctrl/Cmd: distance |
| **Adjust Grid** (P) | horizon middle: tilt · horizon outer 15%: roll around its center · ground: shift grid lines (`grid.offset_*`) | grid yaw | cell size |

Horizon handles use `math3d.horizon_line` (signed pixel distance), `pitch_for_horizon_through` and
`tilt_for_horizon_through`. Drag tests drive real `QMouseEvent`s on a sized, unshown canvas; see
`tests/test_views.py` (`drag`, `grid_canvas`).

## Adding things

- **A new scene parameter:**
  1. Add a field with a default to the relevant dataclass in `model.py`.
  2. If it changes the glyph mask, add it to `RASTER_FIELDS`.
  3. Pass it as a uniform in `render/`.
  4. Expose it by adding one tuple to `SLIDERS` or `COLORS` in `control_panel.py`. Binding and refresh are automatic.
- **A new render pass:** add it to `Compositor.render` so preview and export both get it. Restore GL
  state you change (blend, depth, scissor, alpha-to-coverage, blend equation).
- **A new file:** only when it's a real module boundary. Prefer extending an existing module.

## Coding conventions

- Explicit over clever. Small focused functions, shallow nesting, no one-off helpers or wrappers.
- Match the surrounding comment density. Don't add comments or docstrings to code you didn't change.
- No error handling for impossible cases. Do surface real failures (bad image file, GPU size limits)
  as `ValueError`/`OSError`, and have the views show them in a `QMessageBox`.
- The Model stays Qt-free, `render/` stays widget-free, and views hold no math or GL.

## Testing

- **Every feature ships with tests**: happy path, boundary conditions and failure modes.
- Don't mock the filesystem or OpenGL. Use `tmp_path` and the real offscreen context.
- Fixtures (`tests/conftest.py`): `qapp` (session `QApplication`, OpenGL configured) and
  `gl_context` (a real 4.1 core context on a `QOffscreenSurface`; skips the test if unavailable).
- Render tests assert on actual exported pixels (`render_document`). Compare against a control
  render rather than hard-coding pixel values.
- `test_interactive_frame_rate_exceeds_30_fps` guards the 30+ FPS target. Keep it passing.
- Widgets can be built in tests without being shown. Check visibility with `isVisibleTo(parent)`.
- Never let tests touch the user's real settings. Build `Preferences` on a temporary INI file
  (`QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)`), as the `preferences`
  fixture in `tests/test_views.py` does.
- Modal dialogs (`QFileDialog`, `QMessageBox`) block a test run. Monkeypatch them on the
  `text3d_studio.views.main_window` module. This is the one acceptable kind of mock.
- **Visual check:** to confirm a UI or render change by eye, script a `MainWindow`, load an image,
  then `window.grab().save(...)` and `save_image(window.canvas.render_export(), ...)` from a
  `QTimer.singleShot`, and look at the PNGs.

## Git workflow

- `main` is always releasable. Work on a branch per feature/fix: `feature/<name>`, `fix/<name>`,
  `docs/<name>`.
- Granular commits: one logical change each, with a **concise one-line, imperative** message
  (e.g. `Add triangle buffer for wall geometry`). No bullet-list bodies. Tests pass before merge.
- Integrate via PR (or `git merge --no-ff`) so feature boundaries stay visible in history.
- Multi-PR features use an **integration branch** (e.g. `perspective-grid-overlay`). Each PR branches
  from it and targets it, and the integration branch is merged to `main` once the feature is complete.
  A branch can't be named `<integration>/<x>` because git refs collide, so use `feature/<x>`.
- Run pytest with `set -o pipefail` when piping output (e.g. into `tail`). Otherwise a failing
  run can still let the following `git commit` go through.
- **Versioning (SemVer, pre-1.0):** after a release-worthy merge, commit `Bump version to X.Y.Z`
  on `main`. Update **both** `pyproject.toml` and `text3d_studio/__init__.py`, then run `uv lock`.
  Then create an annotated tag `vX.Y.Z` with a one-line summary.
  Minor = features, patch = fixes.
- Never force-push `main`, never amend published commits, never skip hooks (`--no-verify`).
  Ask before pushing unless the user has explicitly asked for a push.

Release history: v0.1.0 core · v0.2.0 render pipeline · v0.2.1 side-wall shadows · v0.3.0 desktop
UI · v0.3.1 UI polish · v0.4.0 adaptive layer density + docs · v0.5.0 image opening
(empty-state prompt, toolbar, drag and drop, Open Recent, size notice).

## Known limitations / next work

- Text is a single line. The background must fit `GL_MAX_TEXTURE_SIZE`. Faint slice banding is
  possible at near edge-on views.
- Image opening is done (v0.5.0). `MainWindow.open_image_dialog()` asks for a file, and
  `load_image_path(path)` is the single load path used by the menu, toolbar, prompt, drops, Open Recent and the CLI.
  Image decoding is synchronous (busy cursor only). Moving it to a worker thread is a possible follow-up.
- Roadmap (README): perspective solver from drawn lines, MSDF glyphs, true mesh extrusion mode,
  multiple text objects with undo, tiled export, packaging.
