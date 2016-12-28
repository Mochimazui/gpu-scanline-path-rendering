
#ifndef _MOCHIMAZUI_COLOR_H_
#define _MOCHIMAZUI_COLOR_H_

#include <cmath>
#include <cstdint>

#include <vector>

namespace Mochimazui {

struct frgba;
struct u8rgba;

struct frgba{
	frgba() {}
	frgba(float _r, float _g, float _b, float _a) 
		:r(_r), g(_g), b(_b), a(_a){}

	frgba &operator = (const u8rgba &);
	//explicit operator u8rgba() const;

	float r;
	float g;
	float b;
	float a;
};

struct u8rgba{
	u8rgba() :r(0), g(0), b(0), a(255) {}
	u8rgba(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a)
		:r(_r), g(_g), b(_b), a(_a){}

	u8rgba &operator = (const frgba &);
	//explicit operator frgba() const;
	explicit operator uint32_t () const { 
		return *((uint32_t*)this); 
	}

	bool operator == (const u8rgba &o) const {
		return (r == o.r) && (g == o.g) && (b == o.b) && (a == o.a);
	}

	bool operator != (const u8rgba &o) const {
		return (r != o.r) || (g != o.g) || (b != o.b) || (a != o.a);
	}

	uint32_t to_rgba_int32() { return 0; }
	uint32_t to_bgra_int32() { return 0; }
	uint32_t to_argb_int32() { return 0; }
	uint32_t to_abgr_int32() { return 0; }

	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
};

// -------- -------- -------- -------- -------- -------- -------- --------
inline frgba& frgba::operator=(const u8rgba &i) {
	r = i.r / 255.f;
	g = i.g / 255.f;
	b = i.b / 255.f;
	a = i.a / 255.f;
	return *this;
}

//inline frgba::operator u8rgba() const {
//}

// -------- -------- -------- -------- -------- -------- -------- --------
inline u8rgba& u8rgba::operator=(const frgba &i) {
	r = (uint8_t)(i.r * 255);
	g = (uint8_t)(i.g * 255);
	b = (uint8_t)(i.b * 255);
	a = (uint8_t)(i.a * 255);
	return *this;
}

inline std::vector<frgba> genHSVColorList(int n) {

	// See https://en.wikipedia.org/wiki/HSL_and_HSV
	// HSV: hue H ¡Ê [0¡ã, 360¡ã], saturation S ¡Ê [0, 1], and value V ¡Ê [0, 1]
	// use V = S = 1;

	std::vector<frgba> cl;
	float delta = 6.f / n;
	for (int i = 0; i < n; ++i) {
		float h = delta * i;
		float x = 1.f - std::abs(std::fmod(h, 2.f) - 1.f);

		float c = 1.0f;
		if ((0 <= h) && (h < 1)) { cl.push_back(frgba(c, x, 0.f, 1.f)); }
		else if ((1 <= h) && (h < 2)) { cl.push_back(frgba(x, c, 0.f, 1.f)); }
		else if ((2 <= h) && (h < 3)) { cl.push_back(frgba(0.f, c, x, 1.f)); }
		else if ((3 <= h) && (h < 4)) { cl.push_back(frgba(0.f, x, c, 1.f)); }
		else if ((4 <= h) && (h < 5)) { cl.push_back(frgba(x, 0.f, c, 1.f)); }
		else if ((5 <= h) && (h < 6)) { cl.push_back(frgba(c, 0.f, x, 1.f)); }
		else { cl.push_back(frgba(0.f, 0.f, 0.f, 1.f)); }			
	}
	return cl;
}

// -------- -------- -------- -------- -------- -------- -------- --------
inline float srgb_to_lrgb(float f) {
	if (f <= 0.04045f)
		return f / 12.92f;
	else
		return powf((f + 0.055f) / (1.f + 0.055f), 2.4f);
}

inline float lrgb_to_srgb(float f) {
	if (f <= 0.0031308f)
		return 12.92f*f;
	else
		return (1.f + 0.055f)*powf(f, 1.f / 2.4f) - 0.055f;
}

// -------- -------- -------- -------- -------- -------- -------- --------
inline frgba srgb_to_lrgb(const frgba &f) {
	frgba o;
	o.r = srgb_to_lrgb(f.r);
	o.g = srgb_to_lrgb(f.g);
	o.b = srgb_to_lrgb(f.b);
	o.a = f.a;
	return o;
}

inline frgba lrgb_to_srgb(const frgba &f) {
	frgba o;
	o.r = lrgb_to_srgb(f.r);
	o.g = lrgb_to_srgb(f.g);
	o.b = lrgb_to_srgb(f.b);
	o.a = f.a;
	return o;
}

// -------- -------- -------- -------- -------- -------- -------- --------
inline u8rgba srgb_to_lrgb(const u8rgba &c) {
	u8rgba o;
	o.r = (uint8_t)(srgb_to_lrgb(c.r / 255.f) * 255.f);
	o.g = (uint8_t)(srgb_to_lrgb(c.g / 255.f) * 255.f);
	o.b = (uint8_t)(srgb_to_lrgb(c.b / 255.f) * 255.f);
	o.a = c.a;
	return o;
}

inline u8rgba lrgb_to_srgb(const u8rgba &c) {
	u8rgba o;
	o.r = (uint8_t)(lrgb_to_srgb(c.r / 255.f) * 255.f);
	o.g = (uint8_t)(lrgb_to_srgb(c.g / 255.f) * 255.f);
	o.b = (uint8_t)(lrgb_to_srgb(c.b / 255.f) * 255.f);
	o.a = c.a;
	return o;
}

// -------- -------- -------- -------- -------- -------- -------- --------
inline frgba lerp(frgba c0, frgba c1, float t) {
	auto __local_lerp = [](float a, float b, float t) { return a * (1 - t) + b*t; };
	return frgba(
		__local_lerp(c0.r, c1.r, t),
		__local_lerp(c0.g, c1.g, t),
		__local_lerp(c0.b, c1.b, t),
		__local_lerp(c0.a, c1.a, t)
		);
}

} // end of namespace Mochimazui

#endif
