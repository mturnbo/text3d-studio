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
out float v_t;
out float v_height;
flat out int v_layer;
void main() {
    int layer = gl_InstanceID;
    float t = u_layers > 1 ? float(layer) / float(u_layers - 1) : 0.0;
    vec4 local = vec4(u_origin + a_pos * u_size, -t * u_depth, 1.0);
    vec4 world = u_model * local;
    v_pos = world.xyz;
    v_t = t;
    v_layer = layer;
    v_uv = vec2(a_pos.x, 1.0 - a_pos.y);
    v_height = (dot(u_ground_plane.xyz, world.xyz) + u_ground_plane.w) / u_world_size;
    gl_Position = u_proj * (u_flatten * world);
}
"""

LAYER_STACK_FS = """
#version 410 core
in vec2 v_uv;
in vec3 v_pos;
in float v_t;
flat in int v_layer;
uniform sampler2D u_mask;
uniform mat4 u_model;
uniform int u_layers;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient;
uniform vec3 u_face_color;
uniform vec3 u_side_color;
uniform float u_side_darkening;
uniform float u_bevel_strength;
uniform float u_specular;
uniform float u_shininess;
out vec4 frag;
void main() {
    // Cap the mip level: grazing-angle layers would otherwise average thin strokes below the 0.5 cutoff.
    float lod = min(textureQueryLod(u_mask, v_uv).y, 1.5);
    vec4 m = textureLod(u_mask, v_uv, lod);
    float edge = max(fwidth(m.a), 1e-4);
    float coverage = clamp((m.a - 0.5) / edge + 0.5, 0.0, 1.0);
    if (coverage <= 0.0) discard;

    vec2 dir = m.rg * 2.0 - 1.0;
    vec3 n_local;
    vec3 base;
    if (v_layer == 0) {
        n_local = normalize(vec3(dir * m.b * u_bevel_strength, 1.0));
        base = u_face_color;
    } else if (v_layer == u_layers - 1) {
        n_local = vec3(0.0, 0.0, -1.0);
        base = u_side_color * (1.0 - u_side_darkening);
    } else {
        n_local = length(dir) > 0.01 ? normalize(vec3(dir, 0.0)) : vec3(0.0, 0.0, 1.0);
        base = u_side_color * (1.0 - u_side_darkening * v_t);
    }

    vec3 n = normalize(mat3(u_model) * n_local);
    vec3 l = normalize(u_light_dir);
    vec3 v = normalize(-v_pos);
    float diffuse = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float spec = diffuse > 0.0 ? pow(max(dot(n, h), 0.0), u_shininess) * u_specular : 0.0;
    vec3 color = base * (u_ambient + u_light_color * diffuse) + u_light_color * spec;
    frag = vec4(color, coverage);
}
"""

SHADOW_FS = """
#version 410 core
in vec2 v_uv;
in float v_height;
uniform sampler2D u_mask;
uniform float u_falloff;
out vec4 frag;
void main() {
    // The flattened quad is heavily foreshortened, so sample full resolution instead of a mip level.
    if (textureLod(u_mask, v_uv, 0.0).a < 0.5) discard;
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
