
#version 450

#define SIZE 10

uniform ivec2 vp_size;

vec2 vs[4] = {
	vec2(0, vp_size.y - 60),
	vec2(300, vp_size.y - 60),
	vec2(300, vp_size.y),
	vec2(0, vp_size.y)
};

void main() {
	vec2 v = vs[gl_VertexID];
	gl_Position = vec4(
		v.x / vp_size.x * 2 - 1,
		v.y / vp_size.y * 2 - 1,
		0, 1 );
}
