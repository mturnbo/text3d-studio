# Text3D Studio

Desktop app (PyQt6 + OpenGL 4.1 core) for compositing extruded 3D text into a photo using the
**layer-stack duplication** technique, with system fonts, perspective matching, directional lighting
and soft ground shadows. Exports at the photo's native resolution.

## Install & run

```bash
uv sync
uv run text3d-studio path/to/photo.jpg   # or a .t3d project; the argument is optional
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
- Pass a path on the command line: `uv run text3d-studio photo.jpg`, or a project: `uv run text3d-studio scene.t3d`

Exports render at the loaded image's size. The status bar shows it, and says so when a new image changes it.

## Looks

The **Looks** menu applies a ready-made combination of the selected text's material with the scene's light
and shadow: **Gold at sunset**, **Chrome at noon**, **Frosted glass, overcast**, **Marble** and **Oak**. A look
is one step to undo. **Save Look…** stores your current material, light and shadow under a name, and it then
appears in the menu; **Delete Look** removes one of yours (the built-in looks stay). Saved looks are small
JSON files in the app's data folder (on macOS `~/Library/Application Support/Text3D Studio/looks`); a damaged
one is skipped and the status bar says so.

## Exporting

- **File → Export…** (⌘E / Ctrl+E) saves PNG or JPG at the photo's full size.
- **File → Export Transparent PNG…** (⇧⌘E / Ctrl+Shift+E) saves only the text and its shadow over
  transparency, for layering in another editor. The shadow keeps its colour with partial transparency. Glass
  text still shows the photo it bends, since that is what glass looks like. PNG only (JPEG has no transparency).
  Every effect comes along: wall and contact shadows, the reflection (under the text), and the text's depth of
  field, blur, grade and grain. Bloom's glow becomes a soft, see-through halo; laid over the photo it looks
  close to, not exactly like, the app's (it slightly darkens what is under the glow instead of only adding
  light). Masked parts are left transparent.
- **File → Batch Export…** (⌥⌘E / Ctrl+Alt+E) writes several sizes (native, 50%, a long edge you choose) in
  several formats (PNG, JPG, transparent PNG) into one folder, named `photo_3d_3000x2000.png`,
  `photo_3d_1500x1000.jpg`, `photo_3d_1500x1000_transparent.png` and so on. Existing files are never replaced:
  a number is added (`photo_3d_3000x2000-2.png`). Smaller sizes are rendered at that size, so shadows and edges
  look the same as in the full-size image.

## Viewing

- **Before/after:** hold **\\** over the canvas, or tick **View → Show Original Photo**, to see the photo
  without the text. It only changes the view, never an export.
- **Zoom:** **View → Zoom In** (⌘= / Ctrl+=), **Zoom Out** (⌘- / Ctrl+-), **Fit** (⌘0 / Ctrl+0) and
  **Actual Size** (⌘1 / Ctrl+1, one photo pixel per screen pixel). The status bar shows the zoom. While zoomed
  in, hold **Space** and drag to pan. Every tool keeps working while zoomed. Very large photos can't be zoomed
  past 6144 pixels on their longer side.

## Saving and opening projects

A project (`.t3d`) saves the whole scene (every text object, the ground, grid, lighting, shadow, camera, wall,
effects and the occlusion mask) with a link to the photo.
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
5. **Match the softness.** Overcast scenes need high **Softness** and low **Opacity**. Hard sunlight needs low softness. Tint the **Shadow color** toward the scene's shadow color, usually a desaturated blue. **Contact shadow** adds a tight, dark shadow right where floor text meets the ground, which seats it even under soft overcast light; **Contact size** sets how far up the text it reaches and so how far it spreads.
6. **Pick a material.** Choose **Metal**, **Plastic**, **Stone**, **Wood**, **Glass** or **Neon** from the **Material** dropdown. It loads a starting point that you can adjust, and choosing it again resets it. **Glass** shows the photo behind it, bent by its surface, and through its front you see its own back and side walls. Raise **Frost** to blur what's behind it and **Tint depth** to deepen the colour in thicker glass. Its shadow is lighter and takes on its tint. **Neon** glows: set the **Glow color** and **Glow** strength; the sides glow dimmer, and with the glow at 0 it is a dark glossy tube. **Sides** gives the side walls and back another material's look (gold letters with stone sides, say); **Same** uses the material itself. **Outline** draws a band of **Outline color** around the face's edge, following the letter shapes, up to the full width of the bevel (raise **Bevel width** for a wider one); **Outline metallic** makes it metal.
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

If the photo really shows that wall, tick **Wall › Wall in the photo (catches shadows)**: shadows then fall on
the wall too (wall text shadows the wall around it, and floor text near the wall casts a shadow that climbs it),
and floor shadows stop at the wall's base instead of running on behind it. A light from behind the wall casts
nothing on it. Without the tick, the wall only holds text, and all shadows fall on the floor. Setting **Mount** back
to **Floor** stands it on the floor where it appears, facing the way the wall does. Moving the wall carries its
text along; adjusting the ground keeps both the wall and its text where they are on screen.

## Putting text behind things

When something in the photo should stand in front of the text (a lamp post, a person, a plant), pick **Mask**
(**M**) and paint over it: the photo shows through the text, its shadow and its effects wherever you paint. Hold
**Option** (Alt) to erase. **Mask › Brush size** and **Feather** (how softly the brush's edge fades) set the
brush, which shows as a circle under the cursor; the painted area is tinted while the tool is active (never in
exports). Each stroke is one undo step, and **Clear Mask** removes it all. The mask belongs to the photo: it is
saved in the project, and opening another photo drops it.

## Finishing effects

The **Effects** group finishes the composite so the text sits in the photo's look. Every effect is off (or
neutral) until you move its slider, and all of them are in exports.
- **Reflection** mirrors floor text in a glossy floor (wet street, polished stone). **Reflection blur** roughens
  the floor, and **Reflection fade** sets how quickly it fades with depth below the floor. Wall text isn't reflected.
- **Bloom** makes bright parts of the text glow softly into their surroundings: neon, and strong highlights.
  **Bloom threshold** sets how bright a part must be to glow, **Bloom radius** how far the glow spreads.
- **Grain** adds film grain to the text only, so it matches a grainy or noisy photo; **Grain size** sets its
  coarseness. The grain is fixed to the image, so it doesn't crawl while you drag, and every export is the same.
- **Blur** softens the text to match a soft or slightly out-of-focus photo.
- **Brightness**, **Contrast**, **Saturation** and **Warmth** grade the text only (the photo is never changed):
  pull saturation down for a faded photo, warm it for golden hour, cool it for blue hour.
- **Focus distance** and **Aperture** give depth of field: text away from the focus distance (in the scene's
  units, measured from the camera) blurs, more the farther it is from it and the wider the aperture. Set the focus
  to the distance of the text that should stay sharp; with two pieces of text, the other one blurs.

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
| **2 – Integration quality** | Text "belongs" in the photo | Vanishing-point / horizon tool that solves Tilt+FOV from two drawn lines; light-direction picker from a shadow the user clicks (done: Match Sun); colour grade, grain, blur and depth of field (done), contact shadows (done), wall shadows, ground reflection and bloom (done); match black point; contact-hardening shadows (PCSS) |
| **3 – Geometry fidelity** | Crisp at any size | MSDF glyph masks (sharp corners when magnified); true triangulated extrusion as an optional mode (`QPainterPath` → earcut), with the layer stack kept as the stylized mode; multi-line text, alignment, tracking and arcs (done) |
| **4 – Editing workflow** | Usable as a real editor | Multiple text objects (done), undo/redo (done), JSON project files (done), occlusion masks painted over the photo so the text can sit behind foreground objects (done) |
| **5 – Scale & ship** | Large photos, distribution | Tiled export beyond GPU limits (>16k px), 16-bit/HDR pipeline, PyInstaller / briefcase bundles with signing, CI on macOS/Windows/Linux |

Known MVP limits:
- The background must fit the GPU's maximum texture size.
- When the camera is almost edge-on to the extrusion, the stacked slices can still show faint banding (see Phase 3).
