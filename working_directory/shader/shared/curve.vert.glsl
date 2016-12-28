
#version 450

uniform ivec2 vp_size;

layout(binding = 0) uniform samplerBuffer tb_vertex;
layout(binding = 1) uniform samplerBuffer tb_color;

flat out vec4 fragColor;

void main() {
	vec4 draw = texelFetch(tb_vertex, gl_VertexID);
	vec2 p = vec2(draw.x / vp_size.x, draw.y / vp_size.y) * 2 - vec2(1.0, 1.0);
	fragColor = texelFetch(tb_color, gl_VertexID);
	gl_Position = vec4(p.x, p.y, 0, 1);
}
