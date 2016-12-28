
#version 450

uniform ivec2 vp_size;
uniform int fps;

#define SIZE 8

layout(location = 0) out vec4 color; 

int fps_mask[3] = {
	0x13CF, // 001 0011 1100 1111
	0x13EF, // 001 0011 1110 1111
	0x79CF // 111 1001 1100 1111
};

int number_mask[10] = {
	0x7B6F, // 111 1011 0110 1111
	0x4924, // 100 1001 0010 0100
	0x73E7, // 111 0011 1110 0111
	0x79E7, // 111 1001 1110 0111
	0x49ED, // 100 1001 1110 1101
	0x79CF, // 111 1001 1100 1111
	0x7BCF, // 111 1011 1100 1111
	0x4927, // 100 1001 0010 0111
	0x7BEF, // 111 1011 1110 1111
	0x79EF  // 111 1001 1110 1111
};

int mask = 0;

bool check(int x, int y) {
	if (y < 0 || y >=(SIZE * 5) || x < 0 || x >=( SIZE * 3)) { 
		return false;
	}
	x /= SIZE;
	y /= SIZE;
	return (mask >> (y * 3 + x) & 1) == 1;
}

void main(){ 

	vec2 pos;

	if (gl_FragCoord.x < SIZE * 2) { discard; }

	pos.x = gl_FragCoord.x - SIZE * 2;
	pos.y = vp_size.y - gl_FragCoord.y - SIZE * 2;

	int char_index = int((pos.x / (SIZE * 4)) + 1);
	pos.x = mod(pos.x, (SIZE * 4));

	if (pos.y < 0 || pos.y > (SIZE * 5) || pos.x < 0 || pos.x > SIZE * 3) { discard; }

	color = vec4(1, 1, 1, 1);

	if (char_index > 4 && char_index <=7) {
		mask = fps_mask[char_index - 5];
	}
	else if(char_index <=3) {
		if (char_index == 1 && fps >= 100) {
			mask = number_mask[fps / 100];
		}
		if (char_index == 2 && fps >= 10) {
			mask = number_mask[(fps / 10) % 10];
		}
		if(char_index == 3) {
			mask = number_mask[fps % 10];
		}
	}
	else {
		mask = 0;
	}

	if (check(int(pos.x), int(pos.y))) {

		bool flag = true;

		for (int dx = -2; dx <= 2; ++dx) {
			for (int dy = -2; dy <= 2; ++dy) {
				flag = flag && check(
					int(pos.x) + dx, int(pos.y) + dy);
			}
		}

		if (!flag) {
			color = vec4(0, 0, 0, 1);
		}
	}
	else {
		discard;
	}
};
