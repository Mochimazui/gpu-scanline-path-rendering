
#version 450

// --------------------------------
//layout(location = 0) in vec2 vertex;

uniform ivec2 vp_size;

// --------------------------------
void main() {

	int width = vp_size.x;
	int height = vp_size.y;

	vec2 v;

	if (gl_VertexID == 0) {
		v = vec2(0.f, 0.f);
	}
	else if (gl_VertexID == 1) {
		v = vec2(0.f, height);
	}
	else if (gl_VertexID == 2) {
		v = vec2(width, height);
	}
	else {
		v = vec2(width, 0.f);
	}

	v.x = v.x / width * 2.0 - 1.0;
	v.y = v.y / height * 2.0 - 1.0;

	gl_Position = vec4(v.xy, 0.5, 1);
}
