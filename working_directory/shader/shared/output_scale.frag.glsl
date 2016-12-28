
#version 450 

layout(binding = 0) uniform sampler2D scale_tex;

in vec2 texcoord;

layout(location = 0) out vec4 color; 

void main(){ 
	if (texcoord.x < 0 || texcoord.y < 0) {
		color = vec4(0, 0, 0, 1);
	}
	color = texture2D(scale_tex, texcoord);
};
