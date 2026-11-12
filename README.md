# Text3D Studio

Desktop app (PyQt6 + OpenGL 4.1 core) for compositing extruded 3D text into a photo using the
**layer-stack duplication** technique, with system fonts, perspective matching, directional lighting
and soft ground shadows. Exports at the photo's native resolution.

## Install & run

```bash
uv sync
uv run text3d-studio path/to/photo.jpg   # the image argument is optional
uv run pytest                            # 47 tests, including GPU render tests
```

Without uv: `pip install "PyQt6>=6.6" "PyOpenGL>=3.1.7" "numpy>=1.26"`, then `python -m text3d_studio`.

## Architecture (MVVM)

```
Model (model.py)            pure dataclasses: Scene{objects[TextObject{text, transform, material}], ground, light, shadow, camera}, Document
   ▲  mutated only via
ViewModel (viewmodel.py)    SceneViewModel.set(section, field, value) → scene_changed / text_changed / background_changed
   ▲  signals                ▲ signals
Views (views/)              MainWindow · ControlPanel (declarative sliders) · ImageCanvas (QOpenGLWidget + mouse)
   │  ImageCanvas owns a GL context and delegates every pixel to ↓
Render (render/)            Compositor → ShadowRenderer + Text3DRenderer   (no Qt widget imports)
Core                        math3d · lighting.LightingEngine · text_raster · image_io
```

**How the 2D canvas and 3D overlay communicate.** There is no separate overlay: the `Compositor`
renders background, shadow and text into one framebuffer, in this order:

1. **Background.** The photo is drawn as a full-screen textured quad inside a letterboxed viewport that keeps the photo's aspect ratio.
2. **Shadow.** The text is flattened onto its ground plane with a planar projection matrix, rendered at half resolution, Gaussian-blurred, then multiplied onto the background.
3. **Text.** The layer stack is drawn with depth testing. The camera's aspect ratio always equals the photo's, so the preview and the native-resolution export frame the scene identically. The exporter renders the same `Compositor` into an 8× MSAA offscreen FBO.

**Layer-stack extrusion.** `text_raster` draws the string with a system `QFont` into a mask whose
channels encode coverage, outward edge direction and a bevel factor. `Text3DRenderer` draws that
mask as one quad **instanced N times**, each instance pushed back by `depth / (N-1)`. This is
Photoshop's "duplicate layer, nudge, repeat", done in a single draw call:

- **Instance 0 (the face)** gets a beveled normal from the mask.
- **Middle instances (the walls)** use the outward edge direction as their normal and darken progressively with depth.
- **The last instance** is the back cap.

N automatically rises with on-screen extrusion length so the slices never separate into bands.

**Shadows.** A pure layer stack casts no shadow when the light runs parallel to the slices. The
shadow pass therefore uses the front and back slices plus side-wall quads traced from the glyph
outline, which is a closed caster for any light angle. Shadow strength falls off with height above
the ground for a natural contact-shadow look.

**Ground and light.** The ground plane has its own pose relative to the camera: tilt, roll, grid
rotation and camera height. The text stands on it and slides along it, so it stays in perspective
wherever you drag it. The light's azimuth and elevation are measured relative to the ground. Tilting
the ground tilts the sun with it, but rotating the grid or the text doesn't move the sun.

## Opening an image

Any of these loads a PNG, JPG, BMP, WebP or TIFF as the background:
- **Open Image…** on the toolbar, in the File menu, or on the canvas prompt (⌘O / Ctrl+O)
- Drag a single image file onto the canvas
- **File → Open Recent** (last 8 images and projects). The Open dialog starts in the folder you last used.
- Pass a path on the command line: `uv run text3d-studio photo.jpg`

Exports render at the loaded image's size. The status bar shows it, and says so when a new image changes it.

## Saving and opening projects

A project (`.t3d`) saves the whole scene (every text object, the ground, grid, lighting, shadow and camera)
with a link to the photo.
- **File → Save Project** (⌘S / Ctrl+S) saves to the current project, or asks where the first time.
  **Save Project As…** (⇧⌘S / Ctrl+Shift+S) saves a copy under a new name.
- **File → Save Project With Photo…** stores a copy of the photo inside the project file, so it opens anywhere
  on its own (the file gets bigger). Later saves of that project keep the photo inside.
- Otherwise the photo is linked: keep the project and photo together (or move the folder that holds both)
  and it is found again. If it can't be found, the scene still opens; the status bar says which photo is
  missing and the canvas offers to open an image in its place. Photos on network shares aren't opened from
  a project file.
- **File → Open Project…** (⇧⌘O / Ctrl+Shift+O), dragging a `.t3d` onto the canvas, or **Open Recent** opens one.
- The window title shows the project's name, with the unsaved-changes marker when there are edits. Closing
  the window or opening a project with unsaved changes asks whether to save them first. Opening a new photo
  keeps your text and just swaps the picture.
  Opening a project starts a fresh undo history.

## Getting a natural look

1. **Turn on the grid** (**G**, or **Show Grid** on the toolbar). It shows the ground the text stands on, the horizon, and orange lanes marking the path the text can slide along.
2. **Match the ground.** Fastest: pick **Fit Rectangle** (**R**) and drag the four handles onto the corners of something rectangular on the floor: a tile, rug, doormat or table top. Set its real size under **Grid › Fit width/depth** first; that sets the scale. Tick **Fit: estimate field of view** if you don't know the lens. Handles turn red if the shape can't be a rectangle. Or do it by hand: pick **Adjust Grid** (**P**) and drag the grid's white horizon line onto the photo's horizon. Drag near either end of it to roll it if the photo is slanted. Right-drag to rotate the grid until its lines run along floor tiles, curbs or walls, and drag the ground to slide the lines onto real seams. The **Ground** panel has the same settings as sliders. The text stays where it is on screen while you adjust. Then set **Field of view** to match the lens: wide-angle photos need 60–80°, telephoto 15–30°. **Camera height** sets how far the eye is above the floor.
3. **Place the text.** Switch back to **Move Text** (**V**) and drag the text to move it; it keeps its size on screen. Hold **Shift** to move only straight across or straight up and down. Hold **Option** (Alt) to slide it along the floor in perspective instead, so it shrinks into the distance. Hold **Command** (Ctrl) and drag, or right-drag, to turn it (**Yaw**) to line up with walls, curbs and other strong lines in the scene. Pinch or scroll to resize it.
4. **Find the sun.** Fastest: pick **Match Sun** (**L**) and click three points on something upright that casts a clear shadow on the floor (a post, a person, a bottle): its **top**, its **base** where it meets the floor, then the **tip of its shadow**. The light's **Azimuth** and **Elevation** are solved from them (match the ground first; the solve uses it). Click again to start over. Or set them by hand: **Azimuth** so the text's shadow points the same way as the photo's shadows, **Elevation** low (10–25°) for long morning or evening shadows, high (60–80°) for midday.
5. **Match the softness.** Overcast scenes need high **Softness** and low **Opacity**. Hard sunlight needs low softness. Tint the **Shadow color** toward the scene's shadow color, usually a desaturated blue.
6. **Pick a material.** Choose **Metal**, **Plastic**, **Stone**, **Wood** or **Glass** from the **Material** dropdown. It loads a starting point that you can adjust, and choosing it again resets it. **Glass** shows the photo behind it, bent by its surface. Raise **Frost** to blur what's behind it and **Tint depth** to deepen the colour in thicker glass. Its shadow is lighter and takes on its tint.
7. **Borrow the scene's colors.** Pick the **Ambient color** from the sky or the dominant bounce color, and set **Light color** warm for golden hour. The **Pick** button next to either colour samples it straight from the photo: click it, then click the photo (**Esc** cancels). Use **Metallic** 1 for metals and 0 for everything else, and raise **Roughness** for matte or brushed surfaces. Leave **Reflect the photo** on so glossy text picks up the scene's colors.
8. **Check depth.** Real signage is rarely deeper than 0.2–0.5× its height. Raise **Depth darkening** to fake ambient occlusion on the walls.

## Text layout

The **Text** field takes several lines: press Return for a new line (Tab moves on to the next control). The
**Text** group also sets:
- **Align**: Left, Center or Right, lining each line up against the widest one.
- **Letter spacing**: extra room between letters, as a fraction of the letter height (negative squeezes them together).
- **Line height**: the distance between lines, as a multiple of the font's normal spacing. Blank lines keep their height.
- **Arc**: bends the text along a circle, from −180° to 180°. Positive arcs bend the middle up like a rainbow,
  negative ones down like a smile; at ±180° the widest line forms a half circle. Several lines bend along
  circles that share a center, and each letter turns to follow its curve.

The objects list names each object by its first line.

## Several pieces of text

The **Objects** list at the top of the panel holds every piece of text in the scene. **Add** puts a new
default text in front of the selected one, **Duplicate** copies the selected one (text, placement and
material) the same way, and **Delete** removes it (there is always at least one). Click a row, or click
the text itself on the canvas, to select it. The Text, Text Placement and Material controls, every drag,
scroll and pinch, and the grid's orange lanes then apply to that object. Clicking inside a letter's hole
or on empty floor doesn't change the selection, so you can still drag the selected text from there. The
ground, lighting, shadow and camera are shared, and adjusting the ground keeps every object in place.

## Text on a wall

The scene has one wall: an upright plane standing on the floor. To put text on a wall in the photo:

1. Match the ground first (step 2 above).
2. Pick **Set Wall** (**W**) and drag along the line where the wall meets the floor. An orange base line and a
   faint grid on the wall show where it is while the tool is active (they never appear in exports). The **Wall**
   panel group has the same settings as sliders (**Facing** and **Distance**).
3. Select the text and set **Text Placement › Mount** to **Wall**. The text moves onto the wall where it appears
   on screen (or onto the wall's base line, if it stood in front of the wall), flat against it and facing out.
4. With **Move Text** (**V**), drag it around the wall; it keeps its size on screen. Hold **Option** (Alt) to slide
   it along the wall at its real size instead. It can't go below the floor. **Wall position** and **Wall height**
   in the panel place it exactly. Yaw doesn't apply to wall text.

Wall text still casts its shadow on the floor (the wall itself doesn't catch shadows). Setting **Mount** back
to **Floor** stands it on the floor where it appears, facing the way the wall does. Moving the wall carries its
text along; adjusting the ground keeps both the wall and its text where they are on screen.

## Undo and redo

**Edit → Undo** (⌘Z / Ctrl+Z) and **Edit → Redo** (⇧⌘Z / Ctrl+Shift+Z or Ctrl+Y) step through the last 100
changes to the scene: any slider, colour or setting, adding, duplicating or deleting text, and every canvas
drag. A whole drag (on the canvas or a slider) undoes in one step, and so does a quick burst on one setting:
a scroll or pinch, holding a number field's arrows, or a run of typing in the Text field. Undo also brings
back the object that was selected. Selecting alone isn't a step. ⌘Z works from the number fields too; inside
the Text field it undoes your typing first. Opening a new photo starts a fresh history.

## Roadmap

| Phase | Goal | Key work |
|---|---|---|
| **1 – MVP (this repo)** | Credible composite + export | Layer-stack extrusion, system fonts, bevel, Blinn-Phong (physically based since v0.7), planar soft shadow, native-res MSAA export, 30+ FPS |
| **2 – Integration quality** | Text "belongs" in the photo | Vanishing-point / horizon tool that solves Tilt+FOV from two drawn lines; light-direction picker from a shadow the user clicks; color grading (match black point, saturation, grain/noise, blur to the photo's depth of field); contact-hardening shadows (PCSS) |
| **3 – Geometry fidelity** | Crisp at any size | MSDF glyph masks (sharp corners when magnified); true triangulated extrusion as an optional mode (`QPainterPath` → earcut), with the layer stack kept as the stylized mode; multi-line text, alignment, tracking and arcs (done) |
| **4 – Editing workflow** | Usable as a real editor | Multiple text objects (done), undo/redo (done), JSON project files (done), occlusion masks painted over the photo so the text can sit behind foreground objects |
| **5 – Scale & ship** | Large photos, distribution | Tiled export beyond GPU limits (>16k px), 16-bit/HDR pipeline, PyInstaller / briefcase bundles with signing, CI on macOS/Windows/Linux |

Known MVP limits:
- The background must fit the GPU's maximum texture size.
- When the camera is almost edge-on to the extrusion, the stacked slices can still show faint banding (see Phase 3).
