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
model.py            Pure dataclasses: Scene{objects: [TextObject{id, text, transform, material}], ground, grid,
                    light, shadow, camera, wall, mask, effects}, Document (+ the occlusion mask). No Qt, no OpenGL.
                    Document.native_size is the export size. FIELD_RANGES: numeric ranges shared by the panel's
                    sliders and project loading (clamped; non-finite → default)
viewmodel.py        SceneViewModel(QObject): the ONLY mutation point.
                      set(section, field, value) / nudge(...) / set_background(pixels, path)
                      "text", "transform" and "material" (OBJECT_SECTIONS) address the SELECTED object
                      (vm.selected, vm.selected_object: editor state); other sections are shared by the scene
                      add_object() / duplicate_object() / delete_object() / select(index)
                      set_mount("floor" | "wall"): moves the selected object, keeping it where it appears
                      undo() / redo() / can_undo / can_redo / begin_edit() / end_edit() / clear_history()
                      load_document(document) / dirty / mark_saved()
                      signals: scene_changed(section, field), text_changed(), background_changed(),
                               objects_changed(), selection_changed(index), history_changed(),
                               dirty_changed(bool), document_loaded()
history.py          History: bounded (100) undo/redo stacks of opaque states, with burst merging. No Qt.
views/              Qt widgets only. Never touch GL state or numpy image math here.
  main_window.py      menus (Edit › Undo/Redo in _build_edit_menu, File › project actions in
                      _build_project_actions, Looks in _build_looks_menu), status bar, file dialogs,
                      unsaved-changes prompt, BatchExportDialog
  control_panel.py    Objects list (Add / Duplicate / Delete) + declarative slider/colour/checkbox layout,
                      two-way bound to the ViewModel; a selection change refreshes every per-object control
  image_canvas.py     QOpenGLWidget: owns the GL context, mouse→ViewModel edits, delegates drawing.
                      Keeps a CPU raster per object id (re-rasterized only when that object's text changes),
                      uploads them to the compositor and uses them to pick objects under the cursor.
                      Holds the active tool (TEXT_TOOL / GRID_TOOL / FIT_TOOL / WALL_TOOL / SUN_TOOL) and the
                      eyedropper, which are editor state, not scene state. FitHandlesOverlay and ToolOverlay (plain
                      child widgets) draw the fit handles, the wall preview and the Match Sun points
                      Also view state: zoom/pan (in _image_rect) and the before/after toggle
render/             OpenGL, no widget imports. Usable from any current GL context.
  gl_utils.py         ShaderProgram, UnitQuad, TriangleBuffer, Texture2D, ColorTarget, RenderTarget
  shaders.py          all GLSL 4.10 sources as strings
  text3d_renderer.py  instanced layer stack (beauty) + shadow casters (caps + side walls), one object at a time;
                      Glyph = one object's mask texture + wall triangles
  shadow_renderer.py  half-res shadow masks (floor, wall, contact; all objects) → separable Gaussian blur →
                      multiply composite
  effects.py          EffectsRenderer: ground reflection, text finishing (depth of field, blur, grade, grain), bloom
  compositor.py       frame orchestration (see *Compositing*). Holds a Glyph per object id: set_glyph(id, raster|None),
                      retain_glyphs(ids); the occlusion mask: set_mask(mask, rect)
  exporter.py         offscreen 8× MSAA render at native size (× scale) → resolve → glReadPixels;
                      transparent export (compose_transparent); batch naming (batch_jobs)
  grid_renderer.py    editor-only overlay: ground grid, text footprint and lanes, horizon,
                      vanishing-point rings. Drawn by ImageCanvas after Compositor.render
math3d.py           row-major numpy 4x4 helpers; to_gl() transposes to column-major float32.
                    Also image-space perspective: project_to_image / image_ray (pixels, origin
                    top-left, y down), intersect_ray_plane, 4-point homography, vanishing points,
                    focal_from_vanishing_points, ground_pose_from_rectangle, ground_angles_from_rotation
materials.py        Material KINDS (metal, plastic, stone, wood, glass, neon) and their PRESETS (every MaterialParams
                    field except `kind`). SceneViewModel.apply_material(kind) sets the kind and loads its preset.
                    SIDE_KINDS, transmits(material), sanitized_material_params (project loading)
ground.py           Ground model B: ground pose (GroundParams) → plane, origin, light frame; text
                    placement on it (text_position, text_model_matrix, world_to_ground, text_distance);
                    the wall (wall_rotation, wall_to_world, world_to_wall, wall_plane, wall_through,
                    reanchor_wall, reanchor_wall_text, mount_on_wall, mount_on_floor, wall_grid_lines)
lighting.py         LightingEngine: (light, ground) → world light dir, colours, ground plane, shadow matrix;
                    solve_sun: azimuth and elevation from a vertical object's top, base and shadow tip
occlusion.py        Occlusion masks: paint_segment (soft round brush, paint or erase), encode/decode (compressed undo snapshots)
text_raster.py      QFont → mask (RG edge dir, B bevel, A coverage) + traced side-wall triangles. Lays out
                    multi-line text (align, letter spacing, line height) and bends it along arcs
image_io.py         QImage ⇄ numpy RGBA (top row first); SUPPORTED_EXTENSIONS for dialogs and drops
preferences.py      QSettings-backed last folder + recent files (images and projects, max 8), and the looks folder
                    (injectable: Preferences(settings, looks_directory)). MainWindow requires one.
presets.py          Looks: material + light + shadow. BUILTIN_LOOKS, apply_look(vm, look) (one undo step),
                    capture_look, user looks as JSON files: load_user_looks / save_user_look / delete_user_look
project_io.py       .t3d project files: save_project(path, document, embed_photo) / load_project(path) -> LoadedProject
app.py              configure_opengl() + main()
```

Data flow: widget → `vm.set()` → `scene_changed` → `canvas.update()` → `paintGL()` →
`Compositor.render(scene, fbo, viewport)`. The preview and the export call **the same**
`Compositor.render`, which is why they match. Keep it that way.

### Rendering pipeline

1. **Background**: full-screen quad. The viewport is letterboxed to the photo's aspect ratio, and the
   projection aspect always equals the photo aspect.
2. **Shadow** (if enabled): every object's casters are flattened onto the ground plane with
   `math3d.planar_shadow_matrix` and rendered at half resolution into one RGBA8 mask with `GL_MAX`
   blending (strength fades with height above the ground, controlled by `shadow.falloff`). Each object
   writes its strength × `1 − shadow_color(shadow, its material)` per channel, so a glass object's shadow
   keeps its own tint next to an opaque one. The mask is blurred twice in each direction, then the
   background is **multiplied** by `1 − opacity × mask`.
3. **Text**: each object's glyph quad is drawn as one instanced call, with `layers` instances. Instance 0 is
   the beveled face, the middle instances are side walls (normal = outward edge direction, darkening
   with depth) and the last instance is the back cap. All objects share the text layer's depth buffer, so
   nearer objects hide farther ones whatever their order. Alpha-to-coverage is on.

The full order, with the compositing passes, is under *Compositing*.

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
- **Ground edits keep every object where it appears.** `SceneViewModel.set("ground", …)` calls
  `ground.reanchor_text` for each object. Its anchor stays on the same camera ray, `size` scales with its
  distance along that ray (so apparent size is kept), and text yaw offsets grid yaw. Expect
  `transform.x/z/rot_y/size` change events (for any object, not only the selected one) after a ground edit.
  An object whose ray misses the new ground is left as is.
- **The wall** (`scene.wall`, `WallParams`) is an upright plane on the ground. Its frame is
  `ground rotation · Ry(wall.angle)` (x along the base line, y = ground up, z = its normal, facing the camera's
  side), and its base line passes `wall.offset` from the point below the camera, so it stands at local
  z = −offset. Wall coordinates are (along, height above the floor). `transform.mount` picks the placement:
  `"floor"` uses `x/z/rot_y`, `"wall"` uses `wall_x/wall_y` and ignores the floor fields (and yaw). Wall text's
  back cap lies on the wall (`T(wall point + normal · depth · size) · wall rotation · S(size)`). Every
  placement function takes the scene's wall (`text_model_matrix(ground, transform, wall)`); pass it
  everywhere, or wall text renders on the floor. Wall text still casts a planar shadow on the floor only.
- **Ground edits keep the wall where it appears too.** `_place_objects_from` re-places the wall first
  (`reanchor_wall`: the plane through the camera and the old base line meets the new ground), then wall text
  along its camera ray onto the new wall (`reanchor_wall_text`). `placement_snapshot()` is
  `(ground, [transforms], wall)`. Wall edits themselves move wall text with the wall (it's attached).
- The default scene (pitch 0, height 1.5, text at z −12, yaw −20) renders pixel-identically to
  v0.5.0's defaults. `tests/test_ground.py` holds the legacy conversion if old values ever need migrating.
- Image arrays are `(H, W, 4) uint8`, top row first. Texture row 0 = v 0, so quads sample `1 - y`.
  Exports flip rows after `glReadPixels`.

### Shading (physically based, since v0.7)

- **Linear light.** Colors are picked in sRGB. `lighting.srgb_to_linear` converts the light, ambient
  and material colors in Python, the text shader does all math in linear, and `linear_to_srgb` encodes
  its output. The background pass is untouched, so the photo stays bit-exact. Convert in exactly one
  place, never in both.
- **Model.** Metallic-roughness microfacet BRDF: GGX distribution, Smith geometry, Schlick Fresnel.
  `f0 = mix(0.04, base, metallic)`. The light is scaled by π, so a white matte face turned to the light
  reaches 1.0.
- **Reflections.** With `light.reflect_photo`, the reflection vector does a sphere-map lookup into the
  background texture, blurred by `roughness × max_lod` via its mipmaps. Without a photo the ambient color
  is reflected instead. The split-sum environment BRDF uses Karis's analytic fit (`environment_brdf`).
- **Procedural textures** (stone, wood) sample `NOISE_GLSL` at `v_local` (text space, em units), so
  patterns are continuous from face to side walls and stick to the text. Relief (bump) only runs on the
  front and back slices. A surface's `pattern` comes from `text3d_renderer.PATTERNS`; varnish is a general clear coat.
- **Surfaces.** `LAYER_STACK_FS` shades from two `Surface` uniforms: `u_face` (the object's material, face color)
  and `u_side` (side walls and back cap). With `material.side_kind == "same"` the side surface is the material
  with its side color, exactly as before; with another kind (`materials.SIDE_KINDS`) it is that kind's preset
  (colors, roughness, metallic, pattern, transmission, glow), so the side color doesn't apply (the panel hides
  it). `text3d_renderer._set_surface` fills one. A preset covers every field, so picking a material resets
  `side_kind`. `materials.transmits` (face or sides glass) decides whether an object needs the backdrop; the
  object's shadow still follows its face kind.
- **Outline** (`outline_width` in bevel widths, 0 = off; `outline_color`, `outline_metallic`). Face only: the
  band is where the mask's bevel factor (B: 1 at the silhouette, 0 one bevel width in) exceeds
  `1 − width`, antialiased with `fwidth(B)` (taken before the discard). So it follows the glyph contour, can't
  be wider than the bevel rim (raise `text.bevel_width`), and widens a little at concave corners, where the
  blurred mask spreads. It mixes albedo and metallic, keeps roughness and the pattern underneath, is opaque
  on glass, and doesn't glow. Width 0 skips the branch, so it renders exactly as without.
- **Emission** (`emission_color`, `emission_strength`; the `neon` kind's preset glows, every other preset has
  strength 0, but any kind honours it). Unlit, added in linear light after shading; the side surface glows
  `text3d_renderer.SIDE_GLOW` (0.35) as strongly. Values above 1 are **not preserved**: `linear_to_srgb`
  clamps per channel and the text layer is RGBA8, so a strong glow saturates (and drifts toward white) in
  `text_layer.color`. A bloom pass can only find glow as near-saturated layer pixels; keeping HDR values would
  need a float text layer or a separate emission output.
- **Baked noise** (`text3d_renderer.BAKED_NOISE`, on). `NoiseVolume` renders `gradient_noise` once per renderer
  into a 128³ R16F `GL_REPEAT` 3D texture (`shaders.NOISE_PERIOD` 16 cells, `NOISE_TEXELS_PER_CELL` 8; the
  bake wraps lattice cells with `WRAP_LATTICE`, so it tiles; ~8 ms). Texel k holds the noise at its center,
  (k + 0.5)/8 cells, so trilinear filtering reproduces it there. Only `fbm`'s octaves read it (`octave_noise`);
  direct `gradient_noise` calls (stone speckle and fine bump, wood streaks) stay procedural, because a
  finite-difference bump over trilinear texels shows the texels as a grid. The shader picks the version
  through `shaders.with_define(LAYER_STACK_FS, "BAKED_NOISE")`; the procedural build is unchanged bit for bit.
  Patterns differ in detail (wrapped lattice) but not in character. Deep stone ~11.1 → ~6.1 ms per 1600×1000
  4× MSAA frame, wood ~8.6 → ~7.1 ms (Apple GPU). Keep it bound on unit 3 for every beauty draw: an unbound
  sampler3D triggers macOS "unloadable texture" warnings.
- **Glass** (kinds in `materials.TRANSMISSION`). Before drawing any text, the compositor blits the
  viewport (background + shadow, MSAA resolved) into the mipmapped `backdrop` texture, once per frame for
  all glass objects. So glass doesn't show the other text objects behind it. The shader samples
  it along the refracted ray (IOR 1.5, `REFRACTION_DEPTH` em behind the surface, projected back to the
  screen), blurs it by roughness (shown as **Frost**), tints it by Beer-Lambert absorption
  (`tint^(path × absorption)`, path = depth along the view), and splits it from the surface reflection by
  Fresnel. The output is still opaque.
- **Glass far walls** (`text3d_renderer.GLASS_FAR_WALLS`, on). After the backdrop is captured,
  `Text3DRenderer.draw_far_walls` copies it into a single-sample target, draws each transmissive object's
  stack there with `u_far_walls` and `GL_GREATER` against depth cleared to 0 (the farthest slice: where the
  view leaves the text; the face is skipped), copies the result back and rebuilds the mipmaps. A glass far
  wall shows the backdrop along the ray bent out of the glass (or reflected, past the critical angle), tinted
  by `tint^(absorption × depth × (1/facing − 1))`: the extra glass a slanted wall adds, nothing for a back cap
  facing the camera (the front already tints by the full thickness). An opaque side kind is shaded normally
  with its normal flipped, so a glass face with stone sides shows stone through it. All glass objects share the
  backdrop, so where two overlap the nearer one can show the farther one's walls.
  `shadow_renderer.shadow_color(shadow, material)` lightens and tints that object's shadow by the transmission
  fraction.
- **Materials.** `material.kind` picks a preset. The panel's Material dropdown calls `apply_material`, and
  re-picking the current kind resets it. `control_panel.MATERIAL_CONTROLS` lists the fields shown per kind,
  and `MATERIAL_LABELS` renames them (glass: Tint / Edge tint / Tint depth / Frost). `metallic` is set by the kind and never shown.
  A new material parameter needs: a MaterialParams field, a value in **every** preset (a test enforces
  this), a slider/colour entry, and the kinds that show it.
- Render tests check *properties* (highlight tint, roughness spread, reflection follows the photo)
  using interior pixels (`text_pixels` erodes antialiased edges), not reference images.

### Compositing (wave 2F)

`Compositor.render` runs every pass, in this order, for the preview and every export alike:

| # | Pass | Where | Off when | Transparent export |
|---|---|---|---|---|
| 1 | Shadow masks: floor; wall (`wall.enabled`, casters flattened onto the wall's plane); contact (floor text flattened straight down, fading within `contact_size` em) | `ShadowRenderer.render_mask`, half res | shadow off, or opacity and contact strength 0 | rendered again by `render_shadow_factor` |
| 2 | Reflection layer: floor text mirrored across the ground (light mirrored too), faded by depth below the ground, blurred by roughness | `EffectsRenderer.render_reflection`, half res | `reflection_strength` 0, no floor text | — |
| 3 | Background | `_draw_background` | — | left out |
| 4 | Shadow composite: each pixel multiplied by the shadow of the surface it shows (floor or wall), plus the contact shadow | `ShadowRenderer.composite_onto` | as 1 | alpha = opacity × strength, colour recovered (`compose_transparent`); wall and contact included |
| 5 | Reflection composite, premultiplied over | `composite_reflection` | as 2 | under the text (`read_reflection`) |
| 6 | Glass backdrop capture + far walls | `_capture_backdrop`, `draw_far_walls` | no transmissive object | glass still refracts the photo |
| 7 | Text layer, resolved (premultiplied, coverage alpha) | `_draw_text_layer` | no text | the text |
| 8 | Text finishing: depth of field → blur → grade + grain, into `Compositor.text_color` | `EffectsRenderer.finish_text_layer` | each pass skipped at its neutral value; all off → `text_color` *is* `text_layer.color` | included (the export reads `text_color`) |
| 9 | Text composite (alpha writes off) | `_composite_text_layer` | no text | — |
| 10 | Bloom: `text_color` above the threshold, half res, blurred, **added** | `render_bloom`, `composite_bloom` | `bloom_strength` 0, no text | added to the text's colour, with its brightest channel as alpha (`read_bloom`): over a photo it is an *over*, not an add, so it darkens the photo under the glow slightly |
| 11 | Occlusion: the photo put back where the mask says it is in front | `_draw_occlusion` | no mask, or one not at the photo's size | alpha × (1 − mask): hides text, shadows, reflection and glow alike |

- **Settings** live in `scene.effects` (`EffectsParams`; reflection, bloom, grain, blur, grade, depth of field) and
  `scene.shadow` (`contact_strength`, `contact_size`). Every one is in `model.FIELD_RANGES`, which the panel's
  sliders and project loading share, and defaults to off or neutral, where the pass is skipped and the frame is
  bit-identical to before (tests guard each).
- **Resolution independence.** Every radius is a fraction of the image height (`REFLECTION_BLUR`, `BLOOM_SPREAD`,
  `TEXT_BLUR`, `DOF_BLUR`, grain size in thousandths), so a scaled export matches the preview.
- **Text finishing** ping-pongs between two full-size RGBA8 targets (`EffectsRenderer.finished`) and never touches
  `text_layer` itself; the layer's depth still goes to the composite (the grid overlay tests against it), so a
  blurred edge beyond the text's own pixels has far depth there.
  - **Depth of field.** `DOF_COC_FS` turns the layer's depth into a blur radius (circle of confusion):
    `aperture × DOF_BLUR × height × min(|1 − focus / distance|, 1)`, distance along the view. Pixels without
    text (depth cleared to 1, which includes a sharp edge's partly covered pixels) take the largest radius of
    text whose blur reaches them, from a 48-sample spiral, but never from text behind them, so out-of-focus
    text spreads past its outline and over farther text, not over nearer text. `DOF_BLUR_FS` is then the
    separable Gaussian with each pixel's own radius. An in-focus object keeps its own sharp edge.
  - **Blur** is the shared separable Gaussian (`_blur`) on the premultiplied layer, so the edge softens into
    transparency (alpha included).
  - **Grade and grain** are one pass (`TEXT_FINISH_FS`) on the straight colour: brightness (gain), contrast
    around mid grey, saturation (Rec. 709 luma), warmth (red/blue gains, `WARMTH_SHIFT`), then monochrome value
    noise on a fixed `GRAIN_SEED` lattice in image pixels. It never changes alpha, and the grain is fixed to the
    image (no shimmer while dragging; exports are reproducible).
- **Bloom** extracts per channel `max(c − threshold · a, 0) / (1 − threshold)` from the finished layer, so it finds
  near-saturated pixels: highlights, and neon, whose glow saturates in the RGBA8 layer (see *Emission*). It glows
  over the photo and the text alike.
- **The occlusion mask** (`Document.mask`, (H, W) uint8 at the photo's size, saved in projects) is painted with
  the Mask tool (**M**; Option/Alt erases; `vm.paint_mask`, one undo step per stroke; `scene.mask` is the brush).
  The canvas uploads only the changed rect. `render_mask_overlay` tints it in the preview only.
- **Cost** (1600×1000, 4× MSAA, Apple GPU): about 2.5 ms with nothing on, about 5 ms with every effect on (depth
  of field is the most expensive, ~1 ms). `test_every_effect_on_stays_above_30_fps` guards it.

### Workflow features (wave 1A)

- **Looks** (`presets.py`). A look is a dict per section (`material`, `light`, `shadow`) of field → value.
  `apply_look` sets only fields the model knows, inside one `begin_edit`/`end_edit`, so it's one undo step and
  touches only the selected object's material and the shared light and shadow. Built-ins are full presets
  (`materials.PRESETS` + overrides, dataclass defaults + overrides), so a new Material/Light/Shadow field
  appears in them automatically. User looks are `{"format": "text3d-studio-look", "version": 1, "name", ...}`
  files in `Preferences.looks_directory()` (AppDataLocation/looks). Loading checks each value against the
  default's type and skips a bad file with a warning (shown in the status bar). Names compare case-insensitively;
  a built-in's name is refused, a user look's asks to replace. The Looks menu reads the folder only when it
  opens (`aboutToShow`), so building a MainWindow never touches it; tests pass a `tmp_path` folder.
- **Transparent export** (`render_document(..., transparent=True)`). After the normal render, it reads
  `text_layer.color` (premultiplied) and renders the shadow twice into an RGBA16F target with
  `Compositor.render_shadow_factor`: once colored, once `strength_only` (every caster darkens all channels).
  On white, these give `1 − opacity·strength·(1 − color)` and `1 − opacity·strength`, from which
  `compose_transparent` recovers alpha = opacity × strength and the per-pixel shadow color (glass tints
  included), then puts the text over it as straight alpha. Glass still refracts the photo and its shadow
  (that is what it shows); the photo itself isn't drawn. Over a photo, alpha-over only approximates the
  multiplied shadow (the photo's own color doesn't show through). PNG only: the UI refuses other extensions.
- **Scaled and batch export.** `render_document(..., scale=s)` renders at `scaled_size(native, s)`; the scene is
  resolution independent (projection follows the aspect, shadow softness is relative to image height), so a
  half-size export matches a downscaled native one. `batch_jobs` names files `<photo>_3d_<w>x<h>.png|.jpg` and
  `..._transparent.png`, exports a repeated size once, and never replaces a file: it adds `-2`, `-3`, ….
  `MainWindow.run_batch_export(folder, scales, formats)` renders once per (size, transparent) and stops at
  the first failure, saying how many files were written.
- **Before/after.** Holding backslash on the focused canvas (or View › Show Original Photo) renders a copy of
  the scene with no objects and skips the grid and fit handles. Auto-repeat is ignored and losing focus ends
  the hold. Preview only.
- **Zoom and pan.** `_image_rect(scale)` is the one mapping between widget pixels (origin top left) and the
  image; zoom (`_zoom`, device pixels per image pixel, `None` = fit) and pan (`_pan`, widget pixels, clamped
  so a zoomed image keeps covering the canvas) live there, so picking, drags, the grid and the fit handles
  follow. `paintGL` flips its y to GL's bottom-left origin (needed once the image is panned). The compositor
  renders the *whole* zoomed image into a viewport larger than the framebuffer (its layers are sized to it),
  so zoom stops at `MAX_ZOOMED_SIDE_PX` (6144) on the longer side. A new photo or project resets to Fit.
  Space + left-drag pans and never opens an undo step.

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
- **Rasterizing is CPU work (~0.2 s; arced text ~0.4 s).** Only `text.*` fields (`viewmodel.RASTER_FIELDS`,
  which a test keeps equal to every `TextParams` field) re-rasterize, and the canvas debounces them (`TEXT_DEBOUNCE_MS`). The canvas
  compares each object's `TextParams` with the one its raster came from, so only changed objects
  re-rasterize; added objects skip the debounce (`objects_changed`). Everything else is a uniform and must
  stay cheap. Never re-rasterize per frame.
- **Object ids key the rasters and glyphs**, not list indices (which shift on delete). The viewmodel never
  reuses an id, so a stale cache entry can't match a new object.
- **PyQt signal arguments.** `QAction.triggered` and `QAbstractButton.clicked` pass a `checked`
  bool. Don't connect them directly to methods with optional parameters; wrap them in a lambda.
- **Never write alpha < 1 into the canvas framebuffer.** Qt composites the QOpenGLWidget as premultiplied
  color, so a pixel brighter than its alpha overflows and *wraps* (a 245 blue shows as 9): bright reflections
  turned into rainbow speckles. The text-layer composite and the grid pass mask alpha writes with
  `glColorMask(..., GL_FALSE)`. Exports never showed it because the exporter forces alpha to 255, so test
  the *preview* path: `test_preview_framebuffer_stays_opaque`.
- **Text layer** (`Compositor.text_layer`, a `LayerTarget`). The text is drawn alone into a multisampled
  layer (at least `MIN_LAYER_SAMPLES`, more if the target has more) cleared to transparent, with
  alpha-to-coverage *and* alpha-to-one, then resolved: color comes out premultiplied and alpha is coverage.
  `LAYER_COMPOSITE_FS` puts it over the target with (ONE, ONE_MINUS_SRC_ALPHA), fetching texels 1:1 with
  the viewport. Exports match the old direct render to within one level. Passes that work on the text
  alone read `text_layer.color` / `.depth`; what comes after the finishing passes (the composite, bloom, the
  transparent export) reads `Compositor.text_color` instead.
- **Depth pre-pass** (`text3d_renderer.DEPTH_PREPASS`). The stack is drawn once into depth only
  (`LAYER_DEPTH_FS`), then shaded with `GL_EQUAL` and depth writes off, so expensive shading runs about
  once per pixel instead of once per slice. Real gains are modest on Apple GPUs (~11% without MSAA, ~0
  with it). The dominant cost is multisampled side-wall pixels shaded for several slices, so keep expensive
  per-pixel work (e.g. bump) off side-wall slices.
  - Both passes must compute identical depth and coverage: `LAYER_STACK_VS` declares
    `invariant gl_Position`, and both fragment shaders share `MASK_COVERAGE`. Any new per-slice
    geometry or coverage logic must go into both.
  - `test_depth_prepass_matches_single_pass_rendering` guards bit-exact equality with the one-pass
    path.
- `ShaderProgram` caches uniform locations. Setting a uniform the shader optimized away is a
  harmless no-op (location −1).

### Text layout (text_raster.py)

- **Lines.** `content` is split on `\n`; line *k*'s baseline sits `k × lineSpacing × line_height` below the
  first. Blank lines keep their height, but the mask is cropped to the ink, so leading and trailing blank lines
  (and a trailing newline) change nothing, and the last line's ink bottom stays at local y=0. Text of only
  blank lines rasterizes to `None`.
- **Alignment** (`align`: `ALIGNMENTS` left/center/right; anything else raises `ValueError`) lines up each
  line's *ink* (`QFontMetricsF.tightBoundingRect`), not its advance, within the widest line's ink. So leading
  and trailing spaces don't indent a line.
- **Letter spacing** is `QFont.setLetterSpacing(AbsoluteSpacing, letter_spacing × font px)`: em fractions,
  negative overlaps letters (walls stay closed because they're traced from the union mask).
- **Arc** (`arc`, degrees). 0 takes the straight path (whole lines drawn with `drawText`, so arc 0 is exactly
  straight text). Otherwise every glyph comes from a `QTextLayout` glyph run (keeps kerning, spacing and font
  fallback) and is drawn alone with `drawGlyphRun`, translated onto its arc and rotated to the tangent. The
  widest line's ink spans `arc` degrees on a circle through its baseline (center below the text for positive
  arcs, so the middle bends up; above for negative). Other lines sit on concentric circles one pitch nearer or
  farther and keep their arc length (letter spacing), so alignment is measured along the arc. A line whose
  radius would reach the center is clamped to a 1 px radius (its glyphs bunch up; only extreme line heights).
- The mask encoding (edge direction, bevel) and the walls are derived from the final composited alpha, so
  rotated and overlapping glyphs need nothing special, and picking (`GlyphRaster.covers`) follows the bent ink.
- **Texture limit.** `MAX_TEXTURE_WIDTH` bounds *both* sides of the mask. The layout's bounds (rotated glyph
  boxes for arcs, plus italic slack) are measured first, and the font size shrinks (`MIN_FONT_PIXEL_SIZE`
  floor) until they fit; em units don't change, so the text's world size stays the same.
- Tests check masks directly: `ink_bands`/`ink_columns` for lines and alignment, `assert_walls_closed` (every
  wall vertex has even degree) for the shadow walls.

### Editor overlays (the perspective grid)

- Overlays are **preview only**. They're drawn in `ImageCanvas.paintGL` *after* `Compositor.render` and
  must never be added to `Compositor.render`, or they'd appear in exports.
  `test_export_is_identical_with_grid_shown_or_hidden` guards this.
- They reuse the composite's depth buffer: depth test on, depth writes off, so the text hides them. The
  text-layer composite writes the layer's resolved depth (`gl_FragDepth`) for exactly this.
  The horizon pass sits at the far plane (z = 0.999999) for the same reason.
- Procedural lines use `fwidth` for constant pixel width (scaled by the device pixel ratio). With
  `fract()`-wrapped coordinates, take the derivative of the *unwrapped* value, or a false line shows
  up at every cell's half-way point.
- Grid cells fade out below ~4 px so they don't shimmer toward the horizon. The accent lanes and the
  horizon don't fade.

### Tools and canvas gestures

| Tool | Left-drag | Right-drag | Wheel |
|---|---|---|---|
| **Move Text** (V) | press: select the object under the cursor (empty ground keeps the selection) · drag: move the selected object on screen at constant on-screen size (size rescales with distance) · Shift: lock to dominant axis · Option/Alt: slide on the ground in perspective (grabbed point stays under the cursor) · Cmd/Ctrl: yaw | text yaw | size (pinch too); Ctrl/Cmd: distance |
| **Adjust Grid** (P) | horizon middle: tilt · horizon outer 15%: roll around its center · ground: shift grid lines (`grid.offset_*`) | grid yaw | cell size |
| **Fit Rectangle** (R) | drag the 4 corner handles onto a rectangle in the photo; the ground is solved live (`ground.fit_ground_to_rectangle`) | — | — |
| **Set Wall** (W) | drag along the wall's foot: the base line runs through the press and the cursor (`ground.wall_through`); from or into the sky does nothing | — | — |
| **Match Sun** (L) | click top, base, shadow tip; the third click sets azimuth + elevation as one undo step (`lighting.solve_sun`), a fourth starts over; problems go to `tool_status` | — | — |

With wall-mounted text selected, Move Text drags it on the wall: plain drag keeps its on-screen size (the anchor
follows the cursor's pixels onto `ground.wall_plane`), Option slides it along the wall (`wall_point_at`), and it
never goes below `wall_y = 0`. Cmd/Ctrl yaw and Cmd/Ctrl+wheel distance don't apply to it.

**Eyedropper.** The panel's **Pick** buttons (`EYEDROPPER_COLORS`: light colour and ambient colour) emit
`eyedropper_requested(section, field)`; `MainWindow` calls `canvas.start_eyedropper`. The next press on the photo
sets the colour to the 5×5 photo-pixel average under it, mapped through `_image_rect`, as one undo step; a press
off the photo keeps it waiting, Escape (`keyPressEvent`) or a tool change cancels it.

`ToolOverlay` draws the wall preview (base line + grid up the wall, `math3d.project_segment` clips segments
behind the camera) and the Match Sun points, clipped to the photo. Like `FitHandlesOverlay` it's a child widget,
so exports can't contain it.

Qt reports macOS's Command key as `ControlModifier` and Option as `AltModifier`. The drag mode is fixed on
press; Shift can be toggled mid-drag. Pinch arrives as a native gesture, caught by `PinchFilter`. Every
gesture acts on the selected object.

**Picking** (`ImageCanvas.object_at`) is CPU only: the cursor ray meets each object's front-face plane
(`ground.text_face_hit`, local em coordinates), `GlyphRaster.covers` tests the mask's alpha there, and the
nearest hit wins. Letter counters and gaps between letters fall through to whatever is behind. Side walls
aren't pickable. With a single object no picking runs (nothing to choose, and no raster is needed). The grid
overlay's footprint, lanes and vanishing points follow the selected object.

**Control panel input:** `WheelGuard` sends the wheel to the panel's scroll bar unless the control under the
pointer has focus (sliders, spin boxes and combos are click/Tab focus only; the Text field's viewport is
guarded too). `LetterShortcuts` keeps a
focused spin box from swallowing G/V/P/R.

**Focus in tests:** use `activate(window, qapp)` (`QApplication.setActiveWindow`) and `setFocus()`, not
`activateWindow()` or clicks; macOS won't hand OS focus to a test window while you're using another app.
`activate` processes events once *before* setting the active window: a late deactivation from the previous
test's window otherwise clears it again, which made focus tests fail depending on timing.

**Tests:** an autouse fixture in `conftest.py` runs `gc.collect()` before each test. Widgets dropped by
earlier tests otherwise die during a later canvas's paint, and destroying a GL widget switches the
current context, giving "Framebuffer incomplete (status 0x0)". Don't override `event()` on the canvas
either; add a filter object.

**Fit Rectangle:**
- Handles start as a ~30%-wide rectangle on the current ground. Its real size (`grid.fit_width/depth`) sets
  the ground's scale (camera height) on the first drag.
- Edge 1→2 becomes the grid's x direction, and grid offsets put lines along the rectangle's edges.
- `grid.fit_estimate_fov` also sets the camera's field of view from the two vanishing points.
- An impossible shape turns the handles red, reports why via `fit_status`, and keeps the last valid fit.

**Drags re-place the text from where they started.** Ground-changing drags call
`SceneViewModel.set_ground_pose(anchor=vm.placement_snapshot(), …)` (ground plus every object's transform, in
list order) with the state from the press, or, for Fit Rectangle, from the start of the tracing session. A pose mid-drag that can't hold the text therefore can't
strand it. Never re-anchor incrementally per mouse move.

Don't draw text with QPainter inside `paintGL`. In the core-profile context the GL paint engine garbles
glyphs. Use a child widget overlay (like `FitHandlesOverlay` and the empty-state prompt).

Horizon handles use `math3d.horizon_line` (signed pixel distance), `pitch_for_horizon_through` and
`tilt_for_horizon_through`. Drag tests drive real `QMouseEvent`s on a sized, unshown canvas; see
`tests/test_views.py` (`drag`, `grid_canvas`); the wall, sun and eyedropper tools are tested in
`tests/test_views_tools.py`, which reuses those helpers.

### Undo history

- **State = snapshot.** An undo state is `(copy.deepcopy(scene), vm.selected)`, not a command. `History`
  keeps up to `HISTORY_LIMIT` (100) states to undo plus the undone ones to redo; any new step clears redo.
- **Every scene-changing entry point records a step.** `set`, `nudge`, `set_ground_pose`, `apply_material`,
  `add_object`, `duplicate_object` and `delete_object` each wrap their work in `begin_edit()` / `end_edit()`.
  A new one must do the same, or its change can't be undone and gets folded into the next step. Selection
  (`select`) isn't a step, but each state carries the selection, so undo restores it.
- **Edits nest.** `begin_edit` snapshots only at depth 0; the outermost `end_edit` records the snapshot, and
  only if the scene changed (`!=` on the dataclasses). So a ground edit that re-places every object, or a
  material preset that sets ten fields, is one step. Undo/redo are refused while an edit is open.
- **Gestures are one step.** `ImageCanvas` opens an edit on the first button press and closes it when the last
  button is released (all tools; it begins after the press's selection). `control_panel.SliderUndoGroups`
  does the same per slider, watching mouse press/release rather than `sliderPressed`, because a click away from
  the handle moves the value before `sliderPressed` fires.
- **Bursts merge.** Outside a gesture, `set` passes `end_edit(merge_key=(section, field[, object id]))`.
  `History.record` merges a step into the previous one when the key matches and less than `MERGE_WINDOW_S`
  (0.6 s) passed since the last change: wheel and pinch scaling, spin-box arrow repeats, typing in the Text
  field. Undo/redo or a keyless step ends a burst. The clock is injectable: `SceneViewModel(clock=…)`; tests
  use a fake one instead of sleeping.
- **Restoring** swaps the Scene's fields in place (`vm.scene` stays the same object), records nothing, and
  emits `objects_changed`, `selection_changed`, `scene_changed` for every shared-section field that differs,
  and `text_changed` if any text did, so every view refreshes (the canvas re-rasterizes changed text).
- `set_background` isn't undoable and clears the history; so does `load_document` (opening a project).
- **Shortcuts.** Spin boxes and combo boxes claim Cmd+Z for their own text undo; `control_panel.UndoShortcuts`
  leaves that claim unaccepted so the Edit menu's shortcut runs (like `LetterShortcuts`). The Text field
  (a `QPlainTextEdit`, object name `text.content`, no wrapping, Tab moves focus) keeps its native undo, which
  steps back a typed line and its line break separately. The objects list shows each object's first non-blank
  line.

## Adding things

- **A new scene parameter:**
  1. Add a field with a default to the relevant dataclass in `model.py`.
  2. If it changes the glyph mask, add it to `RASTER_FIELDS`.
  3. Pass it as a uniform in `render/`.
  4. Expose it by adding one tuple to `SLIDERS` or `COLORS` in `control_panel.py`. Binding, refresh and undo
     are automatic.
- **A new scene-changing ViewModel method:** wrap it in `begin_edit()` / `end_edit()` so it records an undo
  step (see *Undo history*).
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
- Modal dialogs (`QFileDialog`, `QMessageBox`, `QInputDialog`, `BatchExportDialog.exec`) block a test run.
  Monkeypatch them on the `text3d_studio.views.main_window` module. This is the one acceptable kind of mock.
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
(empty-state prompt, toolbar, drag and drop, Open Recent, size notice) · v0.6.0 perspective grid
(ground model B, grid overlay, Move Text / Adjust Grid / Fit Rectangle tools) · v0.7.0 materials
(PBR shading, metal / plastic / stone / wood / glass, procedural textures, glass refraction) · v0.7.1 input
fixes (constant-size text drag with Shift/Option/Cmd modes, pinch, panel wheel scrolling, G from number fields) ·
v0.8.0 foundations (text layer, multiple text objects, undo/redo, project save and open) · v0.9.0 workflow and
layout (looks, transparent/batch export, zoom, multi-line and arc text, wall mounting, Match Sun, eyedropper).

### Project files (.t3d)

- **Format.** UTF-8 JSON: `{"format": "text3d-studio-project", "version": 1, "scene": dataclasses.asdict(scene),
  "photo": …}`. `photo` is `null`, `{"relative_path", "absolute_path"}` (relative to the project's folder,
  tried first) or `{"embedded_png": base64}` (**Save Project With Photo…**, or a photo that has no path because
  it came embedded). Save Project reuses the mode the project was last saved or opened with.
- **Versioning rule.** `project_io.FORMAT_VERSION` is the schema version. Adding a field with a default needs
  nothing (loading fills defaults and ignores unknown fields). For an *incompatible* change (rename, move,
  new meaning), bump `FORMAT_VERSION` and add a migration in `load_project` that upgrades older dicts before
  they're read. Files newer than `FORMAT_VERSION` are refused with a `ValueError`.
- **Loading is explicit per dataclass** (`_scene_from_dict`, `_object_from_dict`, `_params_from_dict`).
  Colour fields are listed in `COLOR_FIELDS` and turned back into tuples; a test fails if a new `Color` field
  isn't listed. Object ids are kept (renumbered only if missing or duplicated). Corrupt JSON, a non-project
  file or a damaged scene raise `ValueError`; a missing photo is only a warning (`LoadedProject.warnings`,
  `missing_photo`), shown in the status bar and on the empty-state overlay.
- **Photo path hardening.** A shared project can name any path, so `is_allowed_photo_path(path, windows=…)`
  allowlists the stored and resolved paths before any filesystem call: on Windows only drive-letter paths
  (`C:\…`; no UNC, `\\?\`, `\\.\` or `\??\` prefixes, no driveless rooted paths); on POSIX absolute paths
  except `//…` and the `/net` and `/Network` automount roots (case-insensitive). `relative_path` must be
  relative (no root or drive), but may lead outside the project folder (`../photos/x.jpg`). Only
  `image_io.SUPPORTED_EXTENSIONS` are opened. A rejected path is treated like a missing photo.
- **Saving is atomic**: JSON is streamed into a temp file in the same folder, fsynced, then `os.replace`d; on
  any failure the temp file is removed and the old project is untouched.
- **Dirty tracking.** `vm.dirty` is set by `scene_changed`, `objects_changed` and `background_changed`, and
  cleared by `mark_saved()` and `load_document()`. It isn't tied to the undo history: undoing back to the
  saved state still leaves the window marked modified. `MainWindow.maybe_save_changes()` (Save / Discard /
  Cancel) runs before closing and opening a project. Opening an image doesn't ask: it replaces only the
  photo, keeps the scene, and marks the document dirty. Tests monkeypatch
  `main_window.QMessageBox.question` (the `window` fixtures answer Discard).
- `MainWindow.open_path(path)` dispatches drops and Open Recent by extension to `open_project_path` or
  `load_image_path`.
- Window tests that build a `MainWindow` must sort after `tests/test_render.py` (hence
  `tests/test_views_projects.py`, `tests/test_views_workflow.py`): a window destroyed earlier in the session
  leaves no current GL context for the render fixtures. `tests/test_export.py` (no windows) holds the
  transparent, scaled and batch-naming render tests.

## Known limitations / next work

- The background must fit `GL_MAX_TEXTURE_SIZE`. Faint slice banding is
  possible at near edge-on views.
- Multiple text objects (wave 0.2): glass doesn't refract other text objects (only other glass objects' far
  walls, where they overlap). New and
  duplicated objects are placed in front of the selected one (`ground.placement_in_front`), assuming a
  cap-height ink; very tall or descending text can still overlap a little, and near the image's bottom edge
  the new object may be partly cut off.
- Image opening is done (v0.5.0). `MainWindow.open_image_dialog()` asks for a file, and
  `load_image_path(path)` is the single image load path used by the menu, toolbar, prompt, drops and Open Recent
  (both via `open_path`) and the CLI.
  Image decoding is synchronous (busy cursor only). Moving it to a worker thread is a possible follow-up.
- Zoom renders the whole zoomed image each frame, hence the 6144 px cap: Actual Size isn't reachable for
  photos longer than that. Rendering only the visible part (an off-center projection, with the shadow, glass
  backdrop and text layer sized to it) would lift it. While zoomed, glass near the canvas edge can refract
  undefined pixels from outside the framebuffer.
- Roadmap (README): perspective solver from drawn lines, MSDF glyphs, true mesh extrusion mode,
  tiled export, packaging.
