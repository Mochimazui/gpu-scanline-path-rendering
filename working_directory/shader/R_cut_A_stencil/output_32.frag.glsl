
#version 430

// -------- -------- -------- --------
layout(binding = 4) uniform samplerBuffer tex_table;
layout(binding = 5) uniform sampler2D tex_ramp;
layout(binding = 6) uniform sampler2DMS path_frag_tex;

uniform bool enable_srgb_correction;

// -------- -------- -------- --------
flat in vec4 fragment_color;

flat in ivec2 path_frag_pos;

flat in ivec4 pixel_mask;

// -------- -------- -------- -------- -------- -------- -------- --------
in vec3 gradient_coord_0;
in vec3 gradient_coord_1;
flat in vec3 gradient_ramp_coord;
flat in vec3 gradient_focal_point;

// -------- -------- -------- --------
layout(location = 0) out vec4 out_color;

// -------- -------- -------- --------
float safeRcpP(float a) { return a > 1e-6 ? 1 / a : 0.0; }

// -------- -------- -------- -------- -------- -------- -------- --------
float srgb_to_lrgb_f(float f) {
	if (f <= 0.04045f) { return f / 12.92f; }
	else { return pow((f + 0.055f) / (1.f + 0.055f), 2.4f); }
}

vec4 srgb_to_lrgb(vec4 c) {
	return vec4(srgb_to_lrgb_f(c.r), srgb_to_lrgb_f(c.g), srgb_to_lrgb_f(c.b), c.a);
}

// -------- -------- -------- --------
void main() {

	ivec2 in_frag_pos = ivec2(gl_FragCoord.xy) - path_frag_pos;

	if (gradient_ramp_coord.z > 0.0) {
		vec3 gradient_coord = in_frag_pos.y == 0 ? gradient_coord_0 : gradient_coord_1;
		vec2 d = gradient_coord.xy - gradient_focal_point.xy;
		float A = dot(d, d), B = dot(d, gradient_focal_point.xy);
		float c = min(A*safeRcpP(sqrt(B*B + A*gradient_focal_point.z) - B), 1.0);
		out_color = textureLod(tex_ramp, gradient_ramp_coord.xy + vec2(c*gradient_ramp_coord.z, 0.0), 0);
	}
	else {
		out_color = fragment_color;
	}

	ivec2 sub_pixel_index = ivec2(mod(gl_FragCoord, 1) * 2);

	int mask_index = in_frag_pos.x * 2 + in_frag_pos.y;
	
	int count = bitCount(pixel_mask[mask_index]);

	out_color.a *= count / 32.0;
}
