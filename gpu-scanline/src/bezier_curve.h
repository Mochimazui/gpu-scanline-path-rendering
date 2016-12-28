
#ifndef _MOCHIMAZUI_GEOMETRY_H_
#define _MOCHIMAZUI_GEOMETRY_H_

#include <cstdint>
#include <vector>
#include <functional>

#include <glm/vec2.hpp>
#include <glm/mat2x2.hpp>
#include <glm/detail/func_matrix.hpp>
//#include <glm/ext.hpp>

#include <cuda.h>

#include "cutil_math.h"
#include "bezier_curve_type.h"


namespace Mochimazui {

// Ref 
//   [1] https://en.wikipedia.org/wiki/B%C3%A9zier_curve

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// Bezier curve related functions.

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_reverse(float2 &v0, float2&v1, float2 &v2, float2 &v3) {
#ifdef _DEBUG
	printf("empty d_reverse\n");
#endif
}

template<>
__forceinline__ __device__ void d_reverse<CT_Linear>(float2 &v0, float2&v1, float2 &v2, float2 &v3) {
	float2 tmp = v0; v0 = v1; v1 = tmp;
}

template<>
__forceinline__ __device__ void d_reverse<CT_Quadratic>(float2 &v0, float2&v1, float2 &v2, float2 &v3) {
	float2 tmp = v0; v0 = v2; v2 = tmp;
}

template<>
__forceinline__ __device__ void d_reverse<CT_Cubic>(float2 &v0, float2&v1, float2 &v2, float2 &v3) {
	float2 tmp = v0; v0 = v3; v3 = tmp;
	tmp = v1; v1 = v2; v2 = tmp;
}

template<>
__forceinline__ __device__ void d_reverse<CT_Rational>(float2 &v0, float2&v1, float2 &v2, float2 &v3) {
	float2 tmp = v0; v0 = v2; v2 = tmp;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __host__ __device__ float2 bezier_curve_point(const float2 &v0, const float2 &v1, float t) {
	return lerp2(v0, v1, t);
}

__forceinline__ __host__ __device__ float2 bezier_curve_point(
	const float2 &v0, const float2 &v1, const float2 &v2, float t) {
	float2 lv0 = lerp2(v0, v1, t);
	float2 lv1 = lerp2(v1, v2, t);
	return lerp2(lv0, lv1, t);
}

__forceinline__ __host__ __device__ float2 bezier_curve_point(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float t) {

	float2 qv0 = lerp2(iv0, iv1, t);
	float2 qv1 = lerp2(iv1, iv2, t);
	float2 qv2 = lerp2(iv2, iv3, t);

	float2 lv0 = lerp2(qv0, qv1, t);
	float2 lv1 = lerp2(qv1, qv2, t);

	return lerp2(lv0, lv1, t);
}

__forceinline__ __host__ __device__ float2 bezier_curve_point(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float &iw, float t) {

	float2 lv0 = lerp2(iv0, iv1, t);
	float2 lv1 = lerp2(iv1, iv2, t);
	float2 v = lerp2(lv0, lv1, t);
	float w = (1 - t)*(1 - t) + 2.f*(1 - t)*t*iw + t*t;
	v *= safeRcp(w);
	return v;
}

__forceinline__ __host__ __device__ float2 bezier_curve_point(
	uint8_t curve_type,
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw, float t
	) {

	if (curve_type == CT_Linear) {
		return bezier_curve_point(iv0, iv1, t);
	}
	if (curve_type == CT_Quadratic) {
		return bezier_curve_point(iv0, iv1, iv2, t);
	}
	if (curve_type == CT_Cubic) {
		return bezier_curve_point(iv0, iv1, iv2, iv3, t);
	}
	else {
		return bezier_curve_point(iv0, iv1, iv2, iw, t);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------

template<int CURVE_TYPE>
__forceinline__ __device__ float2 d_curve_point(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw, float t) {
#ifdef _DEBUG
	printf("empty d_curve_point\n");
#endif
	return make_float2(0.f, 0.f);
}


template<>
__forceinline__ __device__ float2 d_curve_point<CT_Linear>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw, float t) {
	return lerp2(iv0, iv1, t);
}

template<>
__forceinline__ __device__ float2 d_curve_point<CT_Quadratic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw, float t) {
	return bezier_curve_point(iv0, iv1, iv2, t);
}


template<>
__forceinline__ __device__ float2 d_curve_point<CT_Cubic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw, float t) {

	return bezier_curve_point(iv0, iv1, iv2, iv3, t);
}

template<>
__forceinline__ __device__ float2 d_curve_point<CT_Rational>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw, float t) {

	return bezier_curve_point(iv0, iv1, iv2, iw, t);
}


// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __host__ __device__ float2 bezier_curve_tangent(
	const float2 &v0, const float2 &v1) {
	return make_float2(v1.x - v0.x, v1.y - v0.y);
}

__forceinline__ __host__ __device__ float2 bezier_curve_tangent(
	const float2 &v0, const float2 &v1, const float2 &v2, float t) {
	auto dxt = 2 * (1 - t)*(v1.x - v0.x) + 2 * t*(v2.x - v1.x);
	auto dyt = 2 * (1 - t)*(v1.y - v0.y) + 2 * t*(v2.y - v1.y);
	return make_float2(dxt, dyt);
}

__forceinline__ __host__ __device__ float2 bezier_curve_tangent(
	const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, float t) {
	auto dxt = 3 * (1 - t)*(1 - t)*(v1.x - v0.x) + 6 * (1 - t)*t*(v2.x - v1.x) + 3 * t*t*(v3.x - v2.x);
	auto dyt = 3 * (1 - t)*(1 - t)*(v1.y - v0.y) + 6 * (1 - t)*t*(v2.y - v1.y) + 3 * t*t*(v3.y - v2.y);

	auto d = sqrt(dxt * dxt + dyt * dyt);

	float delta_t = 1 / 1024.f;
	while (d < 1 / 1024.f && delta_t <= 1) {

		auto t0 = t - delta_t;
		auto t1 = t + delta_t;

		if (t0 < 0.f) { t0 = 0.f; }
		if (t1 > 1.f) { t1 = 1.f; }

		auto p0 = bezier_curve_point(v0, v1, v2, v3, t0);
		auto p1 = bezier_curve_point(v0, v1, v2, v3, t1);

		dxt = p1.x - p0.x;
		dyt = p1.y - p0.y;
		d = sqrt(dxt * dxt + dyt * dyt);

		delta_t *= 2.f;
	}

	return make_float2(dxt, dyt);
}

__forceinline__ __host__ __device__ float2 bezier_curve_tangent(
	const float2 &v0, const float2 &v1, const float2 &v2, float w1, float t) {

	auto t2 = t *t;
	auto omt = 1.f - t;
	auto omt2 = omt * omt;

	auto qxt = omt2 * v0.x + 2 * t*omt*v1.x + t2 * v2.x;
	auto qyt = omt2 * v0.y + 2 * t*omt*v1.y + t2 * v2.y;

	auto dqxt = 2 * omt*(v1.x - v0.x) + 2 * t*(v2.x - v1.x);
	auto dqyt = 2 * omt*(v1.y - v0.y) + 2 * t*(v2.y - v1.y);

	auto wt = omt2 + 2 * t*omt*w1 + t2;
	auto dwt = 2 * omt * (w1 - 1) + 2 * t*(1 - w1);

	auto drxt = dqxt * dwt - qxt * wt / (wt * wt);
	auto dryt = dqyt * dwt - qyt * wt / (wt * wt);

	return make_float2(drxt, dryt);
}

__forceinline__ __host__ __device__ float2 bezier_curve_tangent(
	uint8_t curve_type,
	const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, float w1, float t
	) {

	if (curve_type == CT_Linear) {
		return bezier_curve_tangent(v0, v1);
	}
	else if (curve_type == CT_Quadratic) {
		return bezier_curve_tangent(v0, v1, v2, t);
	}
	else if (curve_type == CT_Cubic) {
		return bezier_curve_tangent(v0, v1, v2, v3, t);
	}
	else {
		return bezier_curve_tangent(v0, v1, v2, w1, t);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

__forceinline__ __host__ __device__ float2 bezier_curve_normal(
	const float2 &v0, const float2 &v1) {
	auto tangent = bezier_curve_tangent(v0, v1);
	return make_float2(tangent.y, -tangent.x);
}

__forceinline__ __host__ __device__ float2 bezier_curve_normal(
	const float2 &v0, const float2 &v1, const float2 &v2, float t) {
	auto tangent = bezier_curve_tangent(v0, v1, v2, t);
	return make_float2(tangent.y, -tangent.x);
}

__forceinline__ __host__ __device__ float2 bezier_curve_normal(
	const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, float t) {
	auto tangent = bezier_curve_tangent(v0, v1, v2, v3, t);
	return make_float2(tangent.y, -tangent.x);
}

__forceinline__ __host__ __device__ float2 bezier_curve_normal(
	const float2 &v0, const float2 &v1, const float2 &v2, float w1, float t) {
	auto tangent = bezier_curve_tangent(v0, v1, v2, w1, t);
	return make_float2(tangent.y, -tangent.x);
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __host__ __device__ float2 offset_bezier_curve_point(
	const float2 &v0, const float2 &v1, float offset, float t) {

	// linear
	// r (t) = v0 * (1-t) + v1 * t;
	//       = v0 * 1 - v0 * t + v1 * t;
	// r'(t) = -v0 + v1;

	auto dxt = -v0.x + v1.x;
	auto dyt = -v0.y + v1.y;

	auto d = sqrt(dxt * dxt + dyt * dyt);
	float2 n = make_float2(dyt / d, -dxt / d);

	return bezier_curve_point(v0, v1, t) + offset * n;
}

__forceinline__ __host__ __device__ float2 offset_bezier_curve_point(
	const float2 &v0, const float2 &v1, const float2 &v2, float offset, float t) {

	// quadratic
	// r (t) = (1-t)^2*v0 + 2*(1-t)*t*v1 + t^2*v2
	// r'(t) = 2*(1-t)(v1-v0)  + 2*t*(v2-v1)

	auto dxt = 2 * (1 - t)*(v1.x - v0.x) + 2 * t*(v2.x - v1.x);
	auto dyt = 2 * (1 - t)*(v1.y - v0.y) + 2 * t*(v2.y - v1.y);

	auto d = sqrt(dxt * dxt + dyt * dyt);
	float2 n = make_float2(dyt / d, -dxt / d);

	return bezier_curve_point(v0, v1, v2, t) + offset * n;
}

__forceinline__ __host__ __device__ float2 offset_bezier_curve_point(
	const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, 
	float offset, float t) {

	// cubic
	// r (t) = ...
	// r'(t) = 3(1-t)^2*(v1-v0) + 6*(1-t)*t*(v2-v1) + 3*t^2*(v3-v2)

	auto dxt = 3 * (1 - t)*(1 - t)*(v1.x - v0.x) + 6 * (1 - t)*t*(v2.x - v1.x) + 3 * t*t*(v3.x - v2.x);
	auto dyt = 3 * (1 - t)*(1 - t)*(v1.y - v0.y) + 6 * (1 - t)*t*(v2.y - v1.y) + 3 * t*t*(v3.y - v2.y);

	auto d = sqrt(dxt * dxt + dyt * dyt);

	float delta_t = 1 / 1024.f;
	while (d < 1/1024.f && delta_t <= 1) {

		auto t0 = t - delta_t;
		auto t1 = t + delta_t;

		if (t0 < 0.f) { t0 = 0.f; }
		if (t1 > 1.f) { t1 = 1.f; }

		auto p0 = bezier_curve_point(v0, v1, v2, v3, t0);
		auto p1 = bezier_curve_point(v0, v1, v2, v3, t1);

		dxt = p1.x - p0.x;
		dyt = p1.y - p0.y;
		d = sqrt(dxt * dxt + dyt * dyt);

		delta_t *= 2.f;
	}

	float2 n = make_float2(dyt / d, -dxt / d);

	return bezier_curve_point(v0, v1, v2, v3, t) + offset * n;
}

__forceinline__ __host__ __device__ float2 offset_bezier_curve_point(
	const float2 &v0, const float2 &v1, const float2 &v2, const float &w1, 
	float offset, float t) {

	// rational
	// r (t) = q(t) / w(t)
	// r'(t) = q'(t)*w(t)  - q(t)*w'(t) / (w(t))^2
	//
	// q (t) = (1-t)^2*v0 + 2*(1-t)*t*v1 + t^2*v2
	// q'(t) = 2*(1-t)(v1-v0)  + 2*t*(v2-v1)
	//
	// w (t) = (1-t)^2*w0 + 2*(1-t)*t*w1 + t^2*w2
	// w'(t) = 2*(1-t)(w1-w0)  + 2*t*(w2-w1)
	//       = 2*(1-t)(w1-1)  + 2*t*(1-w1)

	auto t2 = t *t;
	auto omt = 1.f - t;
	auto omt2 = omt * omt;

	auto qxt = omt2 * v0.x + 2 * t*omt*v1.x + t2 * v2.x;
	auto qyt = omt2 * v0.y + 2 * t*omt*v1.y + t2 * v2.y;

	auto dqxt = 2 * omt*(v1.x - v0.x) + 2 * t*(v2.x - v1.x);
	auto dqyt = 2 * omt*(v1.y - v0.y) + 2 * t*(v2.y - v1.y);

	auto wt = omt2 + 2 * t*omt*w1 + t2;
	auto dwt = 2 * omt * (w1 - 1) + 2 * t*(1 - w1);

	auto rxt = qxt / wt;
	auto ryt = qyt / wt;

	auto drxt = dqxt * dwt - qxt * wt / (wt * wt);
	auto dryt = dqyt * dwt - qyt * wt / (wt * wt);

	auto d = sqrt(drxt * drxt + dryt * dryt);
	float2 n = make_float2(dryt / d, -drxt / d);

	return make_float2(rxt, ryt) + offset * n;
}

__forceinline__ __host__ __device__ float2 offset_bezier_curve_point(
	uint8_t curve_type,
	const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, const float &iw,
	float offset, float t) {

	if (curve_type == CT_Linear) {
		return offset_bezier_curve_point(v0, v1, offset, t);
	}
	else if (curve_type == CT_Quadratic) {
		return offset_bezier_curve_point(v0, v1, v2, offset, t);
	}
	else if (curve_type == CT_Cubic) {
		return offset_bezier_curve_point(v0, v1, v2, v3,offset, t);
	}
	else { //if (curve_type == CT_Rational) {
		return offset_bezier_curve_point(v0, v1, v2, iw, offset, t);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------

template<int CURVE_TYPE>
__forceinline__ __device__ void d_subcurve(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
#ifdef _DEBUG
	printf("empty d_subcurve\n");
#endif
}

template<>
__forceinline__ __device__ void d_subcurve<CT_Linear>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
	ov0 = lerp2(iv0, iv1, t0);
	ov1 = lerp2(iv0, iv1, t1);
}

template<>
__forceinline__ __device__ void d_subcurve<CT_Quadratic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {

	float2 la0 = lerp2(iv0, iv1, t0);
	float2 la1 = lerp2(iv1, iv2, t0);

	float2 lb0 = lerp2(iv0, iv1, t1);
	float2 lb1 = lerp2(iv1, iv2, t1);

	ov0 = lerp2(la0, la1, t0);
	ov1 = lerp2(la0, la1, t1);
	ov2 = lerp2(lb0, lb1, t1);
}

template<>
__forceinline__ __device__ void d_subcurve<CT_Cubic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {

	float a = t1;
	float b = t0 / t1;
	if (b != b) { b = 0.f; }

	// left
	float2 c30 = lerp2(iv0, iv1, a);
	float2 c31 = lerp2(iv1, iv2, a);
	float2 c32 = lerp2(iv2, iv3, a);

	float2 c20 = lerp2(c30, c31, a);
	float2 c21 = lerp2(c31, c32, a);

	ov0 = iv0;
	ov1 = c30;
	ov2 = c20;
	ov3 = lerp2(c20, c21, a);

	// right
	c30 = lerp2(ov0, ov1, b);
	c31 = lerp2(ov1, ov2, b);
	c32 = lerp2(ov2, ov3, b);

	c20 = lerp2(c30, c31, b);
	c21 = lerp2(c31, c32, b);

	ov0 = lerp2(c20, c21, b);
	ov1 = c21;
	ov2 = c32;
	//ov3 = ov3;
}

template<>
__forceinline__ __device__ void d_subcurve<CT_Rational>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {

	auto blossom = [](float2 *B, float Bw, float u, float v, float &w) -> float2
	{
		float uv = u*v;
		float b0 = uv - u - v + 1,
			b1 = u + v - 2 * uv,
			b2 = uv;
		w = 1 * b0 + Bw*b1 + 1 * b2;
		return B[0] * b0 + B[1] * b1 + B[2] * b2;
	};

	float u = t0;
	float v = t1;

	float2 cB[3] = { iv0, iv1, iv2 };
	float cBw = iw;

	float wA, wB, wC;
	float2 A = blossom(cB, cBw, u, u, wA);
	float2 B = blossom(cB, cBw, u, v, wB);
	float2 C = blossom(cB, cBw, v, v, wC);

	float s = 1.0f / sqrt(wA * wC);
	ov1 = s*B;
	ow = s*wB;

	if (u == 0)
	{
		ov0 = cB[0];
		ov2 = C / wC;
	}
	else if (v == 1)
	{
		ov0 = A / wA;
		ov2 = cB[2];
	}
	else
	{
		ov0 = A / wA;
		ov2 = C / wC;
	}
}

__forceinline__ __device__ void d_subcurve(
	uint8_t curve_type,
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {

	if (curve_type == CT_Linear) {
		d_subcurve<CT_Linear>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1, ov2, ov3, ow);
	}
	else if (curve_type == CT_Quadratic) {
		d_subcurve<CT_Quadratic>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1, ov2, ov3, ow);
	}
	else if (curve_type == CT_Cubic) {
		d_subcurve<CT_Cubic>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1, ov2, ov3, ow);
	}
	else { //if (curve_type == CT_Rational) {
		d_subcurve<CT_Rational>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1, ov2, ov3, ow);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

template<int CURVE_TYPE>
__forceinline__ __device__ void d_subcurve(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, uint8_t reversed,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow0, float &ow1, float &ow2) {

	float arcw1 = iw;

	ov0 = iv0;
	ov1 = iv1;
	ov2 = iv2;

	// already done in fetch_vertex.
	//ov1 *= arcw1;

	float2 la0 = lerp2(ov0, ov1, t0);
	float2 la1 = lerp2(ov1, ov2, t0);
	float2 lb0 = lerp2(ov0, ov1, t1);
	float2 lb1 = lerp2(ov1, ov2, t1);
	ov0 = lerp2(la0, la1, t0);
	ov1 = lerp2(la0, la1, t1);
	ov2 = lerp2(lb0, lb1, t1);
	float w0 = (1 - t0)*(1 - t0) + 2.f*(1 - t0)*t0*arcw1 + t0*t0;
	//float w1 = (1 - t0)*(1 - t1) + ((1 - t0)*t1 + (1 - t1)*t0)*arcw1 + t0*t1;
	float w2 = (1 - t1)*(1 - t1) + 2.f*(1 - t1)*t1*arcw1 + t1*t1;

	ov0 *= safeRcp(w0);
	ov2 *= safeRcp(w2);

	if (reversed) {
		float2 t = ov0; ov0 = ov2; ov2 = t;
		float tw = w0; w0 = w2; w2 = tw;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_subcurve(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, uint8_t reversed,
	float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
	d_subcurve<CURVE_TYPE>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1, ov2, ov3, ow);
	if (reversed) { d_reverse<CURVE_TYPE>(ov0, ov1, ov2, ov3); }
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_subcurve_end_point(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, float2 &ov0, float2 &ov1
	) {
#ifdef _DEBUG
	printf("empty d_subcurve_end_point\n");
#endif
}

template<>
__forceinline__ __device__ void d_subcurve_end_point<CT_Linear>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, float2 &ov0, float2 &ov1
	) {
	ov0 = lerp2(iv0, iv1, t0);
	ov1 = lerp2(iv0, iv1, t1);
}

template<>
__forceinline__ __device__ void d_subcurve_end_point<CT_Quadratic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, float2 &ov0, float2 &ov2
	) {
	float2 la0 = lerp2(iv0, iv1, t0);
	float2 la1 = lerp2(iv1, iv2, t0);
	float2 lb0 = lerp2(iv0, iv1, t1);
	float2 lb1 = lerp2(iv1, iv2, t1);
	ov0 = lerp2(la0, la1, t0);
	//ov1 = lerp2(la0, la1, t1);
	ov2 = lerp2(lb0, lb1, t1);
}

template<>
__forceinline__ __device__ void d_subcurve_end_point<CT_Cubic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, float2 &ov0, float2 &ov3
	) {

	float a = t1;
	float b = t0 / t1;
	if (b != b) { b = 0.f; }

	// left
	float2 c30 = lerp2(iv0, iv1, a);
	float2 c31 = lerp2(iv1, iv2, a);
	float2 c32 = lerp2(iv2, iv3, a);

	float2 c20 = lerp2(c30, c31, a);
	float2 c21 = lerp2(c31, c32, a);

	ov0 = iv0;
	float2 ov1 = c30;
	float2 ov2 = c20;
	ov3 = lerp2(c20, c21, a);

	// right
	c30 = lerp2(ov0, ov1, b);
	c31 = lerp2(ov1, ov2, b);
	c32 = lerp2(ov2, ov3, b);

	c20 = lerp2(c30, c31, b);
	c21 = lerp2(c31, c32, b);

	ov0 = lerp2(c20, c21, b);
	//ov1 = c21;
	//ov2 = c32;
	//ov3 = ov3;

}

template<>
__forceinline__ __device__ void d_subcurve_end_point<CT_Rational>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, float2 &ov0, float2 &ov2
	) {

	auto blossom = [](float2 *B, float Bw, float u, float v, float &w) -> float2
	{
		float uv = u*v;
		float b0 = uv - u - v + 1,
			b1 = u + v - 2 * uv,
			b2 = uv;
		w = 1 * b0 + Bw*b1 + 1 * b2;
		return B[0] * b0 + B[1] * b1 + B[2] * b2;
	};

	float u = t0;
	float v = t1;

	float2 cB[3] = { iv0, iv1, iv2 };
	float cBw = iw;

	float wA, wB, wC;
	float2 A = blossom(cB, cBw, u, u, wA);
	float2 B = blossom(cB, cBw, u, v, wB);
	float2 C = blossom(cB, cBw, v, v, wC);

	float s = 1.0f / sqrt(wA * wC);
	//ov1 = s*B;
	//ow = s*wB;

	if (u == 0)
	{
		ov0 = cB[0];
		ov2 = C / wC;
	}
	else if (v == 1)
	{
		ov0 = A / wA;
		ov2 = cB[2];
	}
	else
	{
		ov0 = A / wA;
		ov2 = C / wC;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template <int CURVE_TYPE>
__forceinline__ __device__ void d_subcurve_end_point(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, uint8_t reversed, float2 &ov0, float2 &ov1
	) {
	d_subcurve_end_point<CURVE_TYPE>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1);
	if (reversed) { auto t = ov0; ov0 = ov1; ov1 = t; }
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ void d_subcurve_end_point(int curve_type,
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, float2 &ov0, float2 &ov1
	) {
	switch (curve_type) {
	default: break;
	case CT_Linear:
		d_subcurve_end_point<CT_Linear>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1);
		break;
	case CT_Quadratic:
		d_subcurve_end_point<CT_Quadratic>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1);
		break;
	case CT_Cubic:
		d_subcurve_end_point<CT_Cubic>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1);
		break;
	case CT_Rational:
		d_subcurve_end_point<CT_Rational>(iv0, iv1, iv2, iv3, iw, t0, t1, ov0, ov1);
		break;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ void d_subcurve_end_point(int curve_type,
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, const float &iw,
	const float &t0, const float &t1, uint8_t reversed, float2 &ov0, float2 &ov1
	) {
	switch (curve_type) {
	default: break;
	case CT_Linear:
		d_subcurve_end_point<CT_Linear>(iv0, iv1, iv2, iv3, iw, t0, t1, reversed, ov0, ov1);
		break;
	case CT_Quadratic:
		d_subcurve_end_point<CT_Quadratic>(iv0, iv1, iv2, iv3, iw, t0, t1, reversed, ov0, ov1);
		break;
	case CT_Cubic:
		d_subcurve_end_point<CT_Cubic>(iv0, iv1, iv2, iv3, iw, t0, t1, reversed, ov0, ov1);
		break;
	case CT_Rational:
		d_subcurve_end_point<CT_Rational>(iv0, iv1, iv2, iv3, iw, t0, t1, reversed, ov0, ov1);
		break;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------

__forceinline__ __host__ __device__ float2 d_solve_intersection_pd_t(float2 p0, float2 d0, float2 p2, float2 d2) {

	glm::mat2x2 A;
	A[0][0] = d0.x;
	A[0][1] = d0.y;

	A[1][0] = d2.x;
	A[1][1] = d2.y;

	glm::vec2 Y;
	Y[0] = p2.x - p0.x;
	Y[1] = p2.y - p0.y;

	glm::vec2 X;
	X = glm::inverse(A) * Y;

	return make_float2(X.x, X.y);
}

__forceinline__ __host__ __device__ float2 d_solve_intersection_pd(float2 p0, float2 d0, float2 p2, float2 d2) {

	//auto d0 = _v[1] - _v[0];
	//auto d2 = _v[1] - _v[2];

	//auto p0 = point(0.f);
	//auto p2 = point(1.f);

	// p1 === p0 + t0 * d0 === p2 + t2 * d2;

	// p0.x + t0 * d0.x == p2.x + t2 * d2.x;
	// p0.y + t0 * d0.y == p2.y + t2 * d2.y;

	// t0 * d0.x - t2 * d2.x == p2.x - p0.x
	// t0 * d0.y - t2 * d2.y == p2.y - p0.y;

	//d0 = normalize(d0);
	//d2 = normalize(d2);

	glm::mat2x2 A;
	A[0][0] = d0.x;
	A[0][1] = d0.y;

	A[1][0] = d2.x;
	A[1][1] = d2.y;

	glm::vec2 Y;
	Y[0] = p2.x - p0.x;
	Y[1] = p2.y - p0.y;

	glm::vec2 X;
	X = glm::inverse(A) * Y;

	if (X.x != X.x || X.y != X.y) {
		return make_float2(X.x, X.y);
	}

	float2 p1;
	p1.x = p0.x + X.x * d0.x;
	p1.y = p0.y + X.x * d0.y;

	return p1;
};


// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// approx
template<int CURVE_TYPE>
__forceinline__ __device__ void d_approxOffsetCurve(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float iw, float offset,
	const float2 &v_first, const float2 &v_last,
	uint8_t &o_type, float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
#ifdef _DEBUG
	printf("empty d_approxOffsetCurve\n");
#endif
}

template<>
__forceinline__ __device__ void d_approxOffsetCurve<CT_Linear>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float iw, float offset,
	const float2 &v_first, const float2 &v_last,
	uint8_t &o_type, float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
	ov0 = v_first;
	ov1 = v_last;
	o_type = CT_Linear;
}


template<>
__forceinline__ __device__ void d_approxOffsetCurve<CT_Quadratic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float iw, float offset,
	const float2 &v_first, const float2 &v_last,
	uint8_t &o_type, float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
	auto d0 = iv1 - iv0;
	auto d2 = iv1 - iv2;

	auto p0 = v_first;
	auto p2 = v_last;

	auto p1 = d_solve_intersection_pd(p0, d0, p2, d2);
	if (p1.x != p1.x || p1.y != p1.y) {
		p1 = offset_bezier_curve_point(iv0, iv1, iv2, offset, .5f);
	}

	ov0 = p0;
	ov1 = p1;
	ov2 = p2;
	o_type = CT_Quadratic;
}

template<>
__forceinline__ __device__ void d_approxOffsetCurve<CT_Rational>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float iw, float offset,
	const float2 &v_first, const float2 &v_last,
	uint8_t &o_type, float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {
	d_approxOffsetCurve<CT_Quadratic>( 
		iv0, iv1, iv2, iv3,  iw, offset, v_first, v_last, 
		o_type, ov0, ov1, ov2, ov3, ow);
	ow = iw;
	o_type = CT_Rational;
}

template<>
__forceinline__ __device__ void d_approxOffsetCurve<CT_Cubic>(
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float iw, float offset,
	const float2 &v_first, const float2 &v_last,
	uint8_t &o_type, float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {

	if (iv0 == iv1) {

		auto d0 = iv2 - iv1;
		auto d2 = iv2 - iv3;

		auto p0 = v_first;
		auto p2 = v_last;

		auto p1 = d_solve_intersection_pd(p0, d0, p2, d2);
		if (p1.x != p1.x || p1.y != p1.y) {
			o_type = CT_Linear;
			ov0 = p0;
			ov1 = p2;
		}
		else {
			o_type = CT_Cubic;
			ov0 = p0;
			ov1 = p0;
			ov2 = p1;
			ov3 = p2;
		}

	}
	else if (iv1 == iv2) {

		auto d0 = iv1 - iv0;
		auto d2 = iv2 - iv3;

		auto p0 = v_first;
		auto p2 = v_last;

		auto p1 = d_solve_intersection_pd(p0, d0, p2, d2);
		if (p1.x != p1.x || p1.y != p1.y) {
			o_type = CT_Linear;
			ov0 = p0;
			ov1 = p2;
		}
		else {
			o_type = CT_Cubic;
			ov0 = p0;
			ov1 = p1;
			ov2 = p1;
			ov3 = p2;
		}

	}
	else if (iv2 == iv3) {

		auto d0 = iv1 - iv0;
		auto d2 = iv1 - iv3;

		auto p0 = v_first;
		auto p2 = v_last;

		auto p1 = d_solve_intersection_pd(p0, d0, p2, d2);
		if (p1.x != p1.x || p1.y != p1.y) {
			o_type = CT_Linear;
			ov0 = p0;
			ov1 = p2;
		}
		else {
			o_type = CT_Cubic;
			ov0 = p0;
			ov1 = p1;
			ov2 = p2;
			ov3 = p2;
		}

	}
	else
	{
		
		auto m_v1 = offset_bezier_curve_point(iv1, iv2, offset, 0.f);
		auto m_v2 = offset_bezier_curve_point(iv1, iv2, offset, 1.f);

		//
		auto p0 = v_first;
		auto d0 = iv1 - iv0;

		auto p3 = v_last;
		auto d3 = iv2 - iv3;

		auto p1 = d_solve_intersection_pd(p0, d0, m_v2, m_v1 - m_v2);
		auto p2 = d_solve_intersection_pd(p3, d3, m_v1, m_v2 - m_v1);

		if (p1.x != p1.x || p1.y != p1.y) {
			p1 = m_v1; // mid_line.point(0.f);
		}
		if (p2.x != p2.x || p2.y != p2.y) {
			p2 = m_v2; // mid_line.point(1.f);
		}

		o_type = CT_Cubic;
		ov0 = p0;
		ov1 = p1;
		ov2 = p2;
		ov3 = p3;
	}

}

__forceinline__ __device__ void d_approxOffsetCurve(
	uint8_t curve_type,
	const float2 &iv0, const float2 &iv1, const float2 &iv2, const float2 &iv3, float iw, float offset,
	const float2 &v_first, const float2 &v_last,
	uint8_t &o_type, float2 &ov0, float2 &ov1, float2 &ov2, float2 &ov3, float &ow
	) {

	if (curve_type == CT_Linear) {
		d_approxOffsetCurve<CT_Linear>( iv0, iv1, iv2, iv3, iw, offset, v_first, v_last, o_type, ov0, ov1, ov2, ov3, ow);
	}
	else if (curve_type == CT_Quadratic) {
		d_approxOffsetCurve<CT_Quadratic>(iv0, iv1, iv2, iv3, iw, offset, v_first, v_last, o_type, ov0, ov1, ov2, ov3, ow);
	}
	else if (curve_type == CT_Cubic) {
		d_approxOffsetCurve<CT_Cubic>(iv0, iv1, iv2, iv3, iw, offset, v_first, v_last, o_type, ov0, ov1, ov2, ov3, ow);
	}
	else { //if (curve_type == CT_Rational) {
		d_approxOffsetCurve<CT_Rational>(iv0, iv1, iv2, iv3, iw, offset, v_first, v_last, o_type, ov0, ov1, ov2, ov3, ow);
	}

}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

// 
// @class SimpleBezierCurve
// @brief 'Simple' means it's simple to use.
//

struct SimpleBezierCurve final {

	friend class VGContainer;
public:
	SimpleBezierCurve() {}
	SimpleBezierCurve(const float2 &v0, const float2 &v1, float offset) {
		_type = CT_Linear;
		_v[0] = v0;
		_v[1] = v1;
		_offset = offset;
	}
	SimpleBezierCurve(const float2 &v0, const float2 &v1, const float2 &v2, float offset) {
		_type = CT_Quadratic;
		_v[0] = v0;
		_v[1] = v1;
		_v[2] = v2;
		_offset = offset;
	}
	SimpleBezierCurve(const float2 &v0, const float2 &v1, const float2 &v2, const float2 v3, float offset) {
		_type = CT_Cubic;
		_v[0] = v0;
		_v[1] = v1;
		_v[2] = v2;
		_v[3] = v3;
		_offset = offset;
	}
	SimpleBezierCurve(const float2 &v0, const float2 &v1, const float2 &v2, float w, float offset) {
		_type = CT_Rational;
		_v[0] = v0;
		_v[1] = v1;
		_v[2] = v2;
		_v[3].x = w;
		_offset = offset;
	}
public:
	int type() { return _type; }
	int type() const { return _type; }
	float2 &operator[] (uint32_t i) { return _v[i]; }
	const float2 &operator[] (uint32_t i) const { return _v[i]; }
	float &w() { return _v[3].x; }
	const float &w() const { return _v[3].x; }
	float2 &front() { return _v[0]; }
	const float2 &front() const { return _v[0]; }
	float2 &back() { return _v[(_type & 7) - 1]; }
	const float2 &back() const { return _v[(_type & 7) - 1]; }
	int size() const { return _type & 7; }
public:
	float2 point(float t) const;
	float2 tangent(float t) const;
	float2 normal(float t) const;

	SimpleBezierCurve left(float t) const;
	SimpleBezierCurve right(float t) const;
	SimpleBezierCurve subcurve(float t0, float t1) const;
	SimpleBezierCurve offset(float d) const {
		SimpleBezierCurve new_curve = *this;
		new_curve._offset = d;
		return new_curve;
	}

	SimpleBezierCurve reverse() const {
		SimpleBezierCurve new_curve;
		new_curve._type = _type;
		new_curve._offset = _offset;
		auto vn = new_curve._type & 7;
		for (int i = 0; i < vn; ++i) {
			new_curve._v[i] =  _v[vn - i - 1];
		}
		return new_curve;
	}

public:
	SimpleBezierCurve approxOffsetCurve();
	std::vector<SimpleBezierCurve> segApproxOffsetCurve();

public:
	float line_manhattan_length() const;
	float line_length() const;
	float arc_length() const;
	//float control_point_line_length() const;
public:
	std::vector<SimpleBezierCurve> subdiv_curve(float len) const;
	std::vector<float> subdiv_t(float len) const;
	void subdiv(float len, std::function<void(float, float)>) const;
public:
	bool is_offset_curve()  const {
		union f2i { int i; float f; } fi;
		fi.f = _offset;
		return fi.i != 0;
	}

private:
	SimpleBezierCurve approxOffsetCurve(float2 v_first, float2 v_last);

private:
	int _type = CT_Linear;
	float _offset = 0.f;
	float2 _v[4];
};

static_assert(sizeof(SimpleBezierCurve) == sizeof(float2) * 5, "incorrect SimpleBezierCurve size");

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

template <class PointType, int CURVE_TYPE>
struct BezierCurveControlPoints {
protected:
	PointType control_points[CURVE_TYPE];
};

template <class PointType>
struct BezierCurveControlPoints<PointType, BCT_Rational> {
protected:
	PointType control_points[3];
	decltype(PointType::x) w;
};

// -------- -------- -------- -------- -------- -------- -------- --------
template <int CURVE_TYPE>
struct BezierCurve : public BezierCurveControlPoints<float2, CURVE_TYPE> {
public:
	typedef float2 PointType;
public:
	__host__ __device__ int size() const { return CURVE_TYPE & 7; }

	__host__ __device__ PointType &operator[](uint32_t i) {
		assert(i < size());
		return control_points[i];
	}
	__host__ __device__ const PointType &operator[](uint32_t i) const {
		assert(i < size());
		return control_points[i];
	}

	__host__ __device__ decltype(PointType::x) &w() {
		assert(CURVE_TYPE == BCT_Rational);
		return control_points[(CURVE_TYPE & 7) - 1].x;
	};

	__host__ __device__ const decltype(PointType::x) &w() const {
		assert(CURVE_TYPE == BCT_Rational);
		return control_points[(CURVE_TYPE & 7) - 1].x;
	};
};

static_assert(sizeof(BezierCurve<1>) == sizeof(BezierCurve<1>::PointType), "BezierCurve size error");

// -------- -------- -------- -------- -------- -------- -------- --------
template <int CURVE_TYPE>
struct OffsetBezierCurve {
};

// -------- -------- -------- -------- -------- -------- -------- --------

} // end of namespace Mochimazui

#endif