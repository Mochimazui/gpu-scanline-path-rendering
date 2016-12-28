
#version 450

layout(binding = 0) uniform sampler2DMS tex_msaa;

uniform bool enable_srgb_correction;
uniform int samples;

layout(location = 0) out vec4 out_color;

// -------- -------- -------- -------- -------- -------- -------- --------
float lrgb_to_srgb_f(float f) {
	if (f <= 0.0031308f) { return 12.92f*f; }
	else { return (1.f + 0.055f)*pow(f, 1.f / 2.4f) - 0.055f; }
}

vec4 lrgb_to_srgb(const vec4 c) {
	return vec4(
		lrgb_to_srgb_f(c.r),
		lrgb_to_srgb_f(c.g),
		lrgb_to_srgb_f(c.b),
		c.a
		);
}

// -------- -------- -------- -------- -------- -------- -------- --------
float srgb_to_lrgb_f(float f) {
	if (f <= 0.04045f) { return f / 12.92f; }
	else { return pow((f + 0.055f) / (1.f + 0.055f), 2.4f); }
}

vec4 srgb_to_lrgb(vec4 c) {
	return vec4(
		srgb_to_lrgb_f(c.r),
		srgb_to_lrgb_f(c.g),
		srgb_to_lrgb_f(c.b),
		c.a
		);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void main() {

	ivec2 xy = ivec2( gl_FragCoord.x,  gl_FragCoord.y);
	vec4 acc = vec4(0, 0, 0, 0);

	//if (enable_srgb_correction) {
	if (true) {
	//if (false) {
		for (int i = 0; i<samples; i++) {
			vec4 ci = texelFetch(tex_msaa, xy, i);
			ci = srgb_to_lrgb(ci);
			acc += ci*(1.f / float(samples));
		}

		acc = lrgb_to_srgb(acc);
		out_color = acc;
	}
	else {
		for (int i = 0; i<samples; i++) {
			vec4 ci = texelFetch(tex_msaa, xy, i);
			acc += ci*(1.f / float(samples));
		}
		out_color = acc;
	}
}
