
#version 450

flat in vec4 fragColor;

layout(location = 0) out vec4 color;

void main() {
	color = fragColor;
}
