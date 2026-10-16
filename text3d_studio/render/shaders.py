"""GLSL 4.10 core sources. Kept as plain strings so the renderer has no file-system dependency."""

FULLSCREEN_VS = """
#version 410 core
layout(location = 0) in vec2 a_pos;
uniform bool u_flip_y;
out vec2 v_uv;
void main() {
    v_uv = vec2(a_pos.x, u_flip_y ? 1.0 - a_pos.y : a_pos.y);
    gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

BACKGROUND_FS = """
#version 410 core
in vec2 v_uv;
uniform sampler2D u_image;
uniform bool u_has_image;
out vec4 frag;
void main() {
    if (u_has_image) {
        frag = vec4(texture(u_image, v_uv).rgb, 1.0);
    } else {
        // Neutral checkerboard so the text is visible before a photo is loaded.
        vec2 cell = floor(gl_FragCoord.xy / 24.0);
        float c = mod(cell.x + cell.y, 2.0) < 1.0 ? 0.32 : 0.36;
        frag = vec4(vec3(c), 1.0);
    }
}
"""

# Layer-stack extrusion: one unit quad, instanced once per duplicated layer.
# Instance 0 is the front face; each further instance is pushed back by depth / (layers - 1).
LAYER_STACK_VS = """
#version 410 core
layout(location = 0) in vec2 a_pos;
// The depth pre-pass and the shading pass link this source into different programs; invariance guarantees
// they compute bit-identical depths, so the shading pass can test with GL_EQUAL.
invariant gl_Position;
uniform mat4 u_model;
uniform mat4 u_proj;
uniform mat4 u_flatten;      // identity for the beauty pass, planar shadow projection for the shadow pass
uniform vec2 u_origin;
uniform vec2 u_size;
uniform float u_depth;
uniform int u_layers;
uniform vec4 u_ground_plane;
uniform float u_world_size;
out vec2 v_uv;
out vec3 v_pos;
out vec3 v_local;   // position inside the text in em units (z = depth): procedural textures sample this
out float v_t;
out float v_height;
flat out int v_layer;
void main() {
    int layer = gl_InstanceID;
    float t = u_layers > 1 ? float(layer) / float(u_layers - 1) : 0.0;
    vec4 local = vec4(u_origin + a_pos * u_size, -t * u_depth, 1.0);
    vec4 world = u_model * local;
    v_pos = world.xyz;
    v_local = local.xyz;
    v_t = t;
    v_layer = layer;
    v_uv = vec2(a_pos.x, 1.0 - a_pos.y);
    v_height = (dot(u_ground_plane.xyz, world.xyz) + u_ground_plane.w) / u_world_size;
    gl_Position = u_proj * (u_flatten * world);
}
"""

# 3D gradient noise (Perlin-style, arithmetic hash, no lookup textures) and fractional Brownian motion.
# Procedural materials sample these at the text-local position, so patterns run continuously across the
# face and around the side walls of the layer stack.
NOISE_GLSL = """
vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return -1.0 + 2.0 * fract((p.xxy + p.yxx) * p.zyx);
}

// Roughly in [-1, 1], zero at integer lattice points, smooth (quintic fade) everywhere.
float gradient_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = dot(hash33(i), f);
    float n100 = dot(hash33(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0));
    float n010 = dot(hash33(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0));
    float n110 = dot(hash33(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0));
    float n001 = dot(hash33(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0));
    float n101 = dot(hash33(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0));
    float n011 = dot(hash33(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0));
    float n111 = dot(hash33(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// Octaves of noise at doubling frequency and halving amplitude, normalized back to roughly [-1, 1].
float fbm(vec3 p, int octaves) {
    float sum = 0.0;
    float amplitude = 1.0;
    float total = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amplitude * gradient_noise(p);
        total += amplitude;
        p = p * 2.03 + vec3(17.1, 31.7, 7.3);  // offset so octaves don't share lattice zeros
        amplitude *= 0.5;
    }
    return sum / total;
}
"""

# Shared by the depth pre-pass and the shading pass so both derive exactly the same coverage.
MASK_COVERAGE = """
// Cap the mip level: grazing-angle layers would otherwise average thin strokes below the 0.5 cutoff.
float mask_coverage(sampler2D mask, vec2 uv, out vec4 m) {
    float lod = min(textureQueryLod(mask, uv).y, 1.5);
    m = textureLod(mask, uv, lod);
    float edge = max(fwidth(m.a), 1e-4);
    return clamp((m.a - 0.5) / edge + 0.5, 0.0, 1.0);
}
"""

# Depth pre-pass: finds the visible slice at each pixel (color writes are off).
LAYER_DEPTH_FS = """
#version 410 core
in vec2 v_uv;
uniform sampler2D u_mask;
out vec4 frag;
""" + MASK_COVERAGE + """
void main() {
    vec4 m;
    float coverage = mask_coverage(u_mask, v_uv, m);
    if (coverage <= 0.0) discard;
    frag = vec4(0.0, 0.0, 0.0, coverage);  // alpha drives alpha-to-coverage, as in the shading pass
}
"""

LAYER_STACK_FS = """
#version 410 core
in vec2 v_uv;
in vec3 v_pos;
in vec3 v_local;
in float v_t;
flat in int v_layer;
uniform sampler2D u_mask;
uniform mat4 u_model;
uniform int u_layers;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;   // linear
uniform vec3 u_ambient;       // linear
uniform vec3 u_face_color;    // linear
uniform vec3 u_side_color;    // linear
uniform float u_side_darkening;
uniform float u_bevel_strength;
uniform float u_metallic;
uniform float u_roughness;
uniform sampler2D u_environment;   // the background photo (sRGB), with mipmaps
uniform bool u_has_environment;
uniform float u_environment_lod;   // smallest mip level
uniform float u_environment_strength;
uniform int u_pattern;               // 0 none, 1 stone, 2 wood
uniform vec3 u_secondary_color;      // linear
uniform float u_texture_scale;       // em
uniform float u_texture_strength;
uniform float u_bump_strength;
uniform float u_grain_angle;         // radians
uniform float u_varnish;
uniform bool u_transmissive;         // glass: shows the backdrop, refracted
uniform sampler2D u_backdrop;        // the viewport behind the text (background and shadow), with mipmaps
uniform float u_backdrop_lod;        // smallest mip level
uniform float u_absorption;
uniform mat4 u_proj;
uniform float u_depth;
uniform float u_world_size;
out vec4 frag;

const float PI = 3.14159265359;
const float GLASS_IOR = 1.5;
// How far behind the surface (in em) the refracted ray meets the backdrop. A screen-space stand-in for the
// unknown distance to the scene behind: larger values bend the view through the glass further.
const float REFRACTION_DEPTH = 0.6;
const int PATTERN_STONE = 1;
const int PATTERN_WOOD = 2;

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

// The photo stands in for the surroundings: a sphere-map lookup on the reflection direction, blurred by
// roughness via coarser mip levels. A surface facing the camera reflects the photo's center.
vec3 environment(vec3 dir, float roughness) {
    float m = 2.0 * sqrt(dir.x * dir.x + dir.y * dir.y + (dir.z + 1.0) * (dir.z + 1.0));
    vec2 uv = dir.xy / max(m, 1e-4) + 0.5;
    uv.y = 1.0 - uv.y;  // texture row 0 is the top of the photo
    return srgb_to_linear(textureLod(u_environment, uv, roughness * u_environment_lod).rgb);
}

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Metallic-roughness microfacet shading (GGX distribution, Smith geometry, Schlick Fresnel).
float distribution_ggx(float n_dot_h, float alpha) {
    float a2 = alpha * alpha;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return (n_dot_v / (n_dot_v * (1.0 - k) + k)) * (n_dot_l / (n_dot_l * (1.0 - k) + k));
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// Analytic fit of the pre-integrated environment BRDF (Karis, "Physically Based Shading on Mobile").
vec2 environment_brdf(float n_dot_v, float roughness) {
    vec4 r = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    float a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}
""" + MASK_COVERAGE + NOISE_GLSL + """
// Tilts a normal against a height field's gradient (both in the text's own space).
vec3 perturb(vec3 n, vec3 gradient, float strength) {
    return normalize(n - strength * (gradient - dot(gradient, n) * n));
}

// Fine pitting plus a gentle undulation.
float stone_height(vec3 p) {
    return 0.3 * fbm(p * 1.5, 2) + 0.7 * gradient_noise(p * 22.0) * 0.4;
}

// Granite/marble-like stone: subtle mottling, fine speckle, and thin veins in the secondary color along the
// zero lines of a turbulent, domain-warped field. `p` is in pattern units. `relief` adds bump (see main).
void stone_surface(vec3 p, bool relief, inout vec3 albedo, inout float roughness, inout vec3 n) {
    float mottling = fbm(p * 0.8, 3);
    float speckle = gradient_noise(p * 22.0) + 0.5 * gradient_noise(p * 47.0);
    vec3 warp = vec3(fbm(p * 0.5 + 3.1, 3), fbm(p * 0.5 + 7.7, 3), fbm(p * 0.5 + 1.3, 3));
    float turbulence = fbm(p * 0.6 + 2.0 * warp, 4);
    float vein = 1.0 - smoothstep(0.0, 0.035, abs(turbulence));
    float strength = u_texture_strength;
    albedo = u_face_color * (1.0 + 0.12 * mottling * strength + 0.18 * speckle * strength);
    albedo = mix(albedo, u_secondary_color, clamp(vein * strength + max(-speckle - 0.6, 0.0) * strength, 0.0, 1.0));
    roughness = clamp(roughness + 0.08 * speckle, 0.02, 1.0);
    if (!relief) return;

    const float e = 0.004;
    float h = stone_height(p);
    vec3 gradient = vec3(stone_height(p + vec3(e, 0.0, 0.0)) - h,
                         stone_height(p + vec3(0.0, e, 0.0)) - h,
                         stone_height(p + vec3(0.0, 0.0, e)) - h) / e;
    n = perturb(n, gradient, 0.06 * u_bump_strength);
}

// Growth rings: distance from a trunk axis that runs along the grain, slightly below and behind the text, so
// the face shows gentle arcs. Returns the dark latewood amount; `streak` is the fine fibre grain. `p` in em.
float wood_latewood(vec3 p, out float streak) {
    vec3 axis = vec3(cos(u_grain_angle), sin(u_grain_angle), 0.0);
    vec3 side = vec3(-axis.y, axis.x, 0.0);
    vec3 q = p - (side * -2.5 + vec3(0.0, 0.0, -1.2));
    float along = dot(q, axis);
    vec2 across = vec2(dot(q, side), q.z);
    float distortion = fbm(vec3(along * 0.2, across * 1.2), 3);
    float rings = (length(across) + 0.12 * distortion) / u_texture_scale;
    float ring = fract(rings);
    streak = gradient_noise(vec3(along * 2.0, across * (14.0 / u_texture_scale)))
        + 0.5 * gradient_noise(vec3(along * 5.0, across * (40.0 / u_texture_scale)));
    // Each year differs a little in tone; within a year, earlywood darkens gradually into a thin latewood band.
    float year_tone = 0.25 * gradient_noise(vec3(floor(rings) * 0.37, 0.5, 0.5));
    float latewood = 0.35 * smoothstep(0.2, 0.75, ring) + 0.65 * smoothstep(0.75, 0.9, ring) * (1.0 - smoothstep(0.93, 1.0, ring));
    return clamp(latewood + year_tone, 0.0, 1.0);
}

void wood_surface(vec3 p, bool relief, inout vec3 albedo, inout float roughness, inout vec3 n) {
    float streak;
    float latewood = wood_latewood(p, streak);
    float strength = u_texture_strength;
    albedo = mix(u_face_color, u_secondary_color, clamp(latewood * strength + 0.22 * streak * strength, 0.0, 1.0));
    roughness = clamp(roughness + 0.15 * latewood, 0.02, 1.0);
    if (!relief) return;

    // Latewood stands slightly proud of the softer earlywood.
    const float e = 0.002;
    float unused;
    vec3 gradient = vec3(wood_latewood(p + vec3(e, 0.0, 0.0), unused) - latewood,
                         wood_latewood(p + vec3(0.0, e, 0.0), unused) - latewood,
                         wood_latewood(p + vec3(0.0, 0.0, e), unused) - latewood) / e;
    n = perturb(n, gradient, 0.01 * u_bump_strength);
}

// Reflection off the glass surface: the photo (or ambient) and the light's highlight, both weighted by
// a dielectric's Fresnel term.
vec3 dielectric_reflection(vec3 n, vec3 v, vec3 l, vec3 h, float roughness) {
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_v = max(dot(n, v), 1e-4);
    float specular = distribution_ggx(max(dot(n, h), 0.0), max(roughness * roughness, 1e-3))
        * geometry_smith(n_dot_v, n_dot_l, roughness)
        * fresnel_schlick(max(dot(h, v), 0.0), vec3(0.04)).x / (4.0 * n_dot_v * n_dot_l + 1e-4);
    vec2 env = environment_brdf(n_dot_v, roughness);
    vec3 reflected = u_has_environment
        ? environment(reflect(-v, n), roughness) * u_environment_strength
        : u_ambient;
    return specular * PI * u_light_color * n_dot_l + reflected * (0.04 * env.x + env.y);
}

// Glass: the backdrop seen along the refracted ray, blurred by roughness (frost) and tinted by Beer-Lambert
// absorption over the thickness the ray crosses, plus the surface reflection. Fresnel splits the two, so
// grazing angles reflect more and transmit less.
vec3 glass(vec3 n, vec3 v, vec3 l, vec3 h, vec3 tint, float roughness) {
    vec3 refracted = refract(-v, n, 1.0 / GLASS_IOR);
    vec4 clip = u_proj * vec4(v_pos + refracted * (REFRACTION_DEPTH * u_world_size), 1.0);
    vec2 uv = clamp(clip.xy / clip.w * 0.5 + 0.5, 0.0, 1.0);
    vec3 behind = srgb_to_linear(textureLod(u_backdrop, uv, roughness * u_backdrop_lod).rgb);

    // Thickness along the view: the extrusion depth, longer when looking through the slab at an angle.
    vec3 view_local = normalize(transpose(mat3(u_model)) * v);
    float path = u_depth / max(abs(view_local.z), 0.3);
    vec3 transmittance = pow(max(tint, vec3(1e-3)), vec3(path * u_absorption));

    float fresnel = fresnel_schlick(max(dot(n, v), 0.0), vec3(0.04)).x;
    return behind * transmittance * (1.0 - fresnel) + dielectric_reflection(n, v, l, h, roughness);
}

void main() {
    vec4 m;
    float coverage = mask_coverage(u_mask, v_uv, m);
    if (coverage <= 0.0) discard;

    vec2 dir = m.rg * 2.0 - 1.0;
    vec3 n_local;
    vec3 albedo;
    float shade;  // the layer stack's depth darkening, applied on top of any texture
    if (v_layer == 0) {
        n_local = normalize(vec3(dir * m.b * u_bevel_strength, 1.0));
        albedo = u_face_color;
        shade = 1.0;
    } else if (v_layer == u_layers - 1) {
        n_local = vec3(0.0, 0.0, -1.0);
        albedo = u_side_color;
        shade = 1.0 - u_side_darkening;
    } else {
        n_local = length(dir) > 0.01 ? normalize(vec3(dir, 0.0)) : vec3(0.0, 0.0, 1.0);
        albedo = u_side_color;
        shade = 1.0 - u_side_darkening * v_t;
    }

    float roughness = u_roughness;
    vec3 coat_local = n_local;  // varnish fills the texture's relief, so the coat uses the smooth normal
    // Relief only on the front and back faces: side walls are seen nearly edge-on, where bump barely shows,
    // and with multisampling each wall pixel is shaded for several slices, so skipping it there saves ~22%.
    bool relief = v_layer == 0 || v_layer == u_layers - 1;
    if (u_pattern == PATTERN_STONE) {
        stone_surface(v_local / u_texture_scale, relief, albedo, roughness, n_local);
    } else if (u_pattern == PATTERN_WOOD) {
        wood_surface(v_local, relief, albedo, roughness, n_local);
    }
    vec3 base = albedo * shade;

    vec3 n = normalize(mat3(u_model) * n_local);
    vec3 l = normalize(u_light_dir);
    vec3 v = normalize(-v_pos);
    vec3 h = normalize(l + v);
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_v = max(dot(n, v), 1e-4);
    float alpha = max(roughness * roughness, 1e-3);

    if (u_transmissive) {
        frag = vec4(linear_to_srgb(glass(n, v, l, h, base, roughness)), coverage);
        return;
    }

    vec3 f0 = mix(vec3(0.04), base, u_metallic);  // metals tint their reflection; dielectrics reflect ~4% white
    vec3 f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    vec3 specular = distribution_ggx(max(dot(n, h), 0.0), alpha) * geometry_smith(n_dot_v, n_dot_l, roughness)
        * f / (4.0 * n_dot_v * n_dot_l + 1e-4);
    vec3 diffuse = (1.0 - f) * (1.0 - u_metallic) * base;
    // Light intensity is scaled by PI so a white matte surface facing the light reaches exactly 1.
    vec3 direct = (diffuse + specular * PI) * u_light_color * n_dot_l;

    vec2 env = environment_brdf(n_dot_v, roughness);
    vec3 reflected = u_has_environment
        ? environment(reflect(-v, n), roughness) * u_environment_strength
        : u_ambient;
    vec3 ambient = u_ambient * (1.0 - u_metallic) * base + reflected * (f0 * env.x + env.y);
    vec3 color = direct + ambient;

    if (u_varnish > 0.0) {
        // Clear coat: a smooth dielectric layer over the base. It reflects light and the photo, and dims
        // the base by what it reflects.
        const float coat_roughness = 0.08;
        vec3 nc = normalize(mat3(u_model) * coat_local);
        float nc_dot_l = max(dot(nc, l), 0.0);
        float nc_dot_v = max(dot(nc, v), 1e-4);
        float coat_fresnel = fresnel_schlick(nc_dot_v, vec3(0.04)).x;
        float coat_specular = distribution_ggx(max(dot(nc, h), 0.0), coat_roughness * coat_roughness)
            * geometry_smith(nc_dot_v, nc_dot_l, coat_roughness)
            * fresnel_schlick(max(dot(h, v), 0.0), vec3(0.04)).x / (4.0 * nc_dot_v * nc_dot_l + 1e-4);
        vec2 coat_env = environment_brdf(nc_dot_v, coat_roughness);
        vec3 coat_reflected = u_has_environment
            ? environment(reflect(-v, nc), coat_roughness) * u_environment_strength
            : u_ambient;
        vec3 coat = coat_specular * PI * u_light_color * nc_dot_l + coat_reflected * (0.04 * coat_env.x + coat_env.y);
        color = mix(color, color * (1.0 - coat_fresnel) + coat, u_varnish);
    }
    frag = vec4(linear_to_srgb(color), coverage);
}
"""

# Side walls traced from the glyph outline; only used as shadow casters (see text_raster.extract_walls).
WALL_VS = """
#version 410 core
layout(location = 0) in vec3 a_wall;  // (local x, local y, 0 = front / 1 = back)
uniform mat4 u_model;
uniform mat4 u_proj;
uniform mat4 u_flatten;
uniform float u_depth;
uniform vec4 u_ground_plane;
uniform float u_world_size;
out vec2 v_uv;
out float v_height;
void main() {
    vec4 world = u_model * vec4(a_wall.xy, -a_wall.z * u_depth, 1.0);
    v_uv = vec2(0.0);
    v_height = (dot(u_ground_plane.xyz, world.xyz) + u_ground_plane.w) / u_world_size;
    gl_Position = u_proj * (u_flatten * world);
}
"""

SHADOW_FS = """
#version 410 core
in vec2 v_uv;
in float v_height;
uniform sampler2D u_mask;
uniform bool u_use_mask;
uniform float u_falloff;
out vec4 frag;
void main() {
    // The flattened quad is heavily foreshortened, so sample full resolution instead of a mip level.
    if (u_use_mask && textureLod(u_mask, v_uv, 0.0).a < 0.5) discard;
    // Parts of the text further from the ground cast weaker shadow (blended with MAX).
    frag = vec4(clamp(1.0 - max(v_height, 0.0) * u_falloff, 0.0, 1.0));
}
"""

BLUR_FS = """
#version 410 core
in vec2 v_uv;
uniform sampler2D u_image;
uniform vec2 u_step;
out vec4 frag;
void main() {
    float total = 0.0;
    float weights = 0.0;
    for (int i = -12; i <= 12; ++i) {
        float x = float(i) / 6.0;
        float w = exp(-0.5 * x * x);
        total += texture(u_image, v_uv + u_step * float(i)).r * w;
        weights += w;
    }
    frag = vec4(total / weights);
}
"""

SHADOW_COMPOSITE_FS = """
#version 410 core
in vec2 v_uv;
uniform sampler2D u_image;
uniform vec3 u_color;
uniform float u_opacity;
out vec4 frag;
void main() {
    float s = texture(u_image, v_uv).r * u_opacity;
    frag = vec4(mix(vec3(1.0), u_color, s), 1.0);  // multiplied onto the background
}
"""

# ----- Editor overlay (preview only, never exported) ----------------------------------------------

# One large quad on the ground plane, in ground coordinates.
GRID_VS = """
#version 410 core
layout(location = 0) in vec2 a_pos;
uniform mat4 u_proj;
uniform mat4 u_ground;  // ground (x, 0, z) -> camera space
uniform float u_extent;
out vec2 v_ground;
void main() {
    v_ground = (a_pos * 2.0 - 1.0) * u_extent;
    gl_Position = u_proj * (u_ground * vec4(v_ground.x, 0.0, v_ground.y, 1.0));
}
"""

GRID_FS = """
#version 410 core
in vec2 v_ground;
uniform float u_cell;
uniform vec2 u_offset;
uniform vec3 u_color;
uniform vec3 u_accent;
uniform float u_opacity;
uniform float u_pixel_ratio;
uniform bool u_has_text;
uniform vec2 u_text_xz;
uniform float u_text_yaw;   // radians
uniform vec4 u_footprint;   // x min, x max, z min, z max in the text's own frame on the ground
out vec4 frag;

// Coverage of a line `width` device pixels wide where `value` is zero. The derivative is passed in
// separately because fract() makes derivatives of the wrapped value jump between cells.
float line(float value, float derivative, float width) {
    float px = abs(value) / max(derivative, 1e-6);
    return clamp(width * 0.5 + 0.5 - px, 0.0, 1.0);
}

float line(float value, float width) {
    return line(value, fwidth(value), width);
}

void main() {
    vec2 cell = (v_ground - u_offset) / u_cell;
    vec2 cell_fw = fwidth(cell);
    vec2 to_line = fract(cell + 0.5) - 0.5;
    float grid = max(line(to_line.x, cell_fw.x, u_pixel_ratio), line(to_line.y, cell_fw.y, u_pixel_ratio));
    // Fade cells smaller than a few pixels so the grid doesn't shimmer toward the horizon.
    float spacing_px = 1.0 / max(max(cell_fw.x, cell_fw.y), 1e-6);
    float fade = clamp((spacing_px - 4.0 * u_pixel_ratio) / (8.0 * u_pixel_ratio), 0.0, 1.0);
    float grid_a = grid * fade * u_opacity;

    float fill_a = 0.0;
    float accent_a = 0.0;
    if (u_has_text) {
        vec2 p = v_ground - u_text_xz;
        float c = cos(u_text_yaw);
        float s = sin(u_text_yaw);
        float lx = c * p.x - s * p.y;
        float lz = s * p.x + c * p.y;
        vec4 f = u_footprint;
        float inside = step(f.x, lx) * step(lx, f.y) * step(f.z, lz) * step(lz, f.w);
        float width = 2.0 * u_pixel_ratio;
        // Baseline and the two lanes, extended toward the horizon, plus the footprint's back edge.
        float accent = max(max(line(lx - f.x, width), line(lx - f.y, width)), line(lz - f.w, width));
        accent = max(accent, inside * line(lz - f.z, width));
        accent_a = accent * min(1.0, u_opacity + 0.3);
        fill_a = inside * 0.18 * u_opacity;
    }

    // Premultiplied "over": grid, then footprint fill, then accent lines.
    vec3 rgb = u_color * grid_a;
    float a = grid_a;
    rgb = u_accent * fill_a + rgb * (1.0 - fill_a);
    a = fill_a + a * (1.0 - fill_a);
    rgb = u_accent * accent_a + rgb * (1.0 - accent_a);
    a = accent_a + a * (1.0 - accent_a);
    if (a < 1.0 / 255.0) discard;
    frag = vec4(rgb / a, a);
}
"""

# Full-screen quad at the far plane, so the text in front hides it.
HORIZON_VS = """
#version 410 core
layout(location = 0) in vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos * 2.0 - 1.0, 0.999999, 1.0);
}
"""

HORIZON_FS = """
#version 410 core
uniform vec3 u_up;          // ground normal in camera space
uniform vec4 u_viewport;    // x, y, width, height in device pixels
uniform float u_focal;      // focal length in device pixels
uniform vec3 u_color;
uniform vec3 u_accent;
uniform float u_opacity;
uniform float u_pixel_ratio;
uniform vec2 u_vp0;         // vanishing points in framebuffer coordinates
uniform vec2 u_vp1;
uniform bool u_vp0_valid;
uniform bool u_vp1_valid;
out vec4 frag;

float ring(vec2 center) {
    float r = length(gl_FragCoord.xy - center);
    return clamp(u_pixel_ratio + 0.5 - abs(r - 7.0 * u_pixel_ratio), 0.0, 1.0);
}

void main() {
    // The horizon is where a camera ray runs parallel to the ground.
    vec2 p = gl_FragCoord.xy - u_viewport.xy - 0.5 * u_viewport.zw;
    float s = dot(u_up, vec3(p / u_focal, -1.0));
    float horizon = clamp(0.75 * u_pixel_ratio + 0.5 - abs(s) / max(fwidth(s), 1e-6), 0.0, 1.0);
    float marker = 0.0;
    if (u_vp0_valid) marker = max(marker, ring(u_vp0));
    if (u_vp1_valid) marker = max(marker, ring(u_vp1));
    float a = max(horizon, marker) * min(1.0, u_opacity + 0.3);
    if (a < 1.0 / 255.0) discard;
    frag = vec4(marker > horizon ? u_accent : u_color, a);
}
"""

