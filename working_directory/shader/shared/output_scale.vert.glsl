
#version 450

uniform ivec2 vp_size;

uniform ivec2 vp_translate;
uniform float vp_scale;

uniform mat4x4 o_tmat;

out vec2 texcoord;

vec2 v[4] = {
	vec2(0, 0),
	vec2(0, 1),
	vec2(1, 1),
	vec2(1, 0)
};

void calc_texcoord() {
	vec4 ov = vec4(v[gl_VertexID], 0, 1);

	ov.x *= vp_size.x;
	ov.y *= vp_size.y;

	ov = inverse(o_tmat) * ov;
	ov /= ov.w;

	ov.x /= vp_size.x;
	ov.y /= vp_size.y;

	texcoord = ov.xy;
}

void calc_position() {
	vec2 ov = v[gl_VertexID];
	//ov.y = 1.0 - ov.y;
	gl_Position = vec4(ov * 2 - vec2(1, 1), 0.0, 1.0);
}

void main() {
	calc_texcoord();
	calc_position();
}
