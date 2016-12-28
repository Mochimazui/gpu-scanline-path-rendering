
#version 450

// -------- -------- -------- -------- -------- -------- -------- --------
layout(binding = 0) uniform isamplerBuffer tb_index;
layout(binding = 1) uniform isamplerBuffer tb_span;
layout(binding = 2) uniform isamplerBuffer tb_path_fragment;
layout(binding = 3) uniform isamplerBuffer tb_stencil_mask;

layout(binding = 4) uniform samplerBuffer tex_table;
layout(binding = 5) uniform sampler2D tex_ramp;

// -------- -------- -------- -------- -------- -------- -------- --------
uniform vec3 pid2depth_irampheight;
uniform vec3 inv_proj_rx;
uniform vec3 inv_proj_ry;
uniform vec3 inv_proj_rw;
uniform vec3 inv_proj_rp;
uniform float inv_proj_a;

uniform bool enable_srgb_correction;

uniform ivec2 vp_size;

// -------- -------- -------- -------- -------- -------- -------- --------
//flat out int fragment_type;
flat out vec4 fragment_color;

flat out ivec2 path_frag_pos;

flat out int pixel_mask;

// -------- -------- -------- -------- -------- -------- -------- --------
out vec3 gradient_coord_0;
out vec3 gradient_coord_1;
flat out vec3 gradient_ramp_coord;
flat out vec3 gradient_focal_point;

// -------- -------- -------- -------- -------- -------- -------- --------
vec4 u8rgba2frgba(int c) {
	return vec4(c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, (c >> 24) & 0xFF) / 255.0;
}

// -------- -------- -------- -------- -------- -------- -------- --------
float srgb_to_lrgb_f(float f) {
	if (f <= 0.04045f) { return f / 12.92f; }
	else { return pow((f + 0.055f) / (1.f + 0.055f), 2.4f); }
}

vec4 srgb_to_lrgb(vec4 c) {
	return vec4(srgb_to_lrgb_f(c.r), srgb_to_lrgb_f(c.g), srgb_to_lrgb_f(c.b), c.a);
}

// -------- -------- -------- --------
float safeRcpP(float a) { return a > 1e-6 ? 1 / a : 0.0; }

// -------- -------- -------- --------
void calc_color(int colori, vec2 vertex) {

	if (uint(colori - 1) < uint(0x01000000)) {

		// 1. fetch gradient transform & focal point.
		int path_id = (colori - 1) * 3;
		vec4 word0 = texelFetch(tex_table, path_id);
		vec4 word1 = texelFetch(tex_table, path_id + 1);
		gradient_focal_point = texelFetch(tex_table, path_id + 2).xyz;

		// 2. transform back to object space.
		vec2 vertex_0 = vertex + vec2(1.0, 0.5);
		vec2 vertex_1 = vertex + vec2(1.0, 1.5);

		vec3 rd_0 = inv_proj_rx*vertex_0.x + inv_proj_ry*vertex_0.y + inv_proj_rw;
		vec3 rd_1 = inv_proj_rx*vertex_1.x + inv_proj_ry*vertex_1.y + inv_proj_rw;

		vec3 obj_space_vertex_0 = inv_proj_rp + (inv_proj_a / rd_0.z)*rd_0;
		vec3 obj_space_vertex_1 = inv_proj_rp + (inv_proj_a / rd_1.z)*rd_1;

		// 3. transform to gradient space.
		gradient_coord_0 = vec3(
			obj_space_vertex_0.x*word0.xw +
			obj_space_vertex_0.y*vec2(word0.y, word1.x) +
			vec2(word0.z, word1.y),
			1.0);

		gradient_coord_1 = vec3(
			obj_space_vertex_1.x*word0.xw +
			obj_space_vertex_1.y*vec2(word0.y, word1.x) +
			vec2(word0.z, word1.y),
			1.0);

		// 4. ramp.
		int ramp_coordi = floatBitsToInt(word1.z);
		gradient_ramp_coord = vec3(
			(float(ramp_coordi & 1023) + 0.5)*(1.0 / 1024.0),
			(float(ramp_coordi >> 10) + 0.5)*pid2depth_irampheight.z,
			word1.w);
	}
	else {
		gradient_ramp_coord.z = 0.0;
		vec4 color = u8rgba2frgba(colori);
		fragment_color = enable_srgb_correction ? srgb_to_lrgb(color) : color;
	}
}

// -------- -------- -------- --------
void main() {

	int index = gl_VertexID >> 1;
	int line_vi = gl_VertexID & 1;

	ivec4 draw = texelFetch(tb_index, index);

	path_frag_pos = ivec2(draw.x & 0xFFFF, draw.x >> 16);

	vec2 pos = vec2(
		path_frag_pos.x + line_vi * draw.y,
		path_frag_pos.y
		);

	calc_color(draw.z, pos);

	pos.y += 1;

	pos.x = pos.x / float(vp_size.x) * 2 - 1.0;
	pos.y = pos.y / float(vp_size.y) * 2 - 1.0;

	gl_Position = vec4(pos, 0, 1);

	pixel_mask = (draw.w == 0) ? 0xFFFFFFFF
		: texelFetch(tb_stencil_mask, draw.w - 1).r;
}
