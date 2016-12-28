
#ifndef _MOCHIMAZUI_RASTERIZER_SHARED_DEVICE_H_
#define _MOCHIMAZUI_RASTERIZER_SHARED_DEVICE_H_

#include "../../cutil_math.h"
#include "../../vg_container.h"

namespace Mochimazui {

namespace Rasterizer {

// -------- -------- -------- -------- -------- -------- -------- --------
template <class T>
__forceinline__ __host__ __device__ void swap(T &a, T &b) {
	T t = a; a = b; b = t;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template <class T>
__forceinline__ __host__ __device__ T min(const T &a, const T &b) {
	return a < b ? a : b;
}

template <class T>
__forceinline__ __host__ __device__ T max(const T &a, const T &b) {
	return a > b ? a : b;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __host__ __device__ float2 min(const float2 &a, const float2 &b) {
	return make_float2(min(a.x, b.x), min(a.y, b.y));
}

__forceinline__ __host__ __device__ float2 max(const float2 &a, const float2 &b) {
	return make_float2(max(a.x, b.x), max(a.y, b.y));
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ float f4dot(float4 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template <int L, int U>
__forceinline__ __device__ int32_t limit(int32_t x) {
	if (x < L) { return L; }
	if (x > U) { return U; }
	return x;
}

__forceinline__ __device__ int32_t limit(int32_t x, int32_t l, int32_t u) {
	x = max(x, l);
	x = min(x, u);
	return x;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
template <int BIG_FRAG_SIZE>
__forceinline__ __device__ int bigFragmentMask() {
	return 0x0;
}

template <>
__forceinline__ __device__ int bigFragmentMask<2>() {
	return 0xFFFFFFFE;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int BIG_FRAG_SIZE>
__forceinline__ __device__ int bigFragmentFloor(float v) {
	return __float2int_rd(v) &bigFragmentMask<BIG_FRAG_SIZE>();
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int BIG_FRAG_SIZE>
__forceinline__ __device__ int2 bigFragmentFloor(float2 v) {
	return make_int2(
		bigFragmentFloor<BIG_FRAG_SIZE>(v.x),
		bigFragmentFloor<BIG_FRAG_SIZE>(v.y)
		);
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

// shared helper function.

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ float d_misalign(float v) {
	//return floor(v) != v ? v : v + 1.f / 1024.f;
	return v;
}

__forceinline__ __device__ void d_misalign(float2 & v) {
	v.x = d_misalign(v.x);
	v.y = d_misalign(v.y);
}

template <int CURVE_TYPE>
__forceinline__ __device__ void d_misalign(float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
#ifdef _DEBUG
	printf("empty d_misalign\n");
#endif
}

template <>
__forceinline__ __device__ void d_misalign<CT_Linear>(float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	d_misalign(v0); d_misalign(v1);
}

template <>
__forceinline__ __device__ void d_misalign<CT_Quadratic>(float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	d_misalign(v0); d_misalign(v1); d_misalign(v2);
}

template <>
__forceinline__ __device__ void d_misalign<CT_Cubic>(float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	d_misalign(v0); d_misalign(v1); d_misalign(v2); d_misalign(v3);
}

template <>
__forceinline__ __device__ void d_misalign<CT_Rational>(float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	d_misalign(v0); d_misalign(v1); d_misalign(v2);
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_fetchVertex(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w) {
#ifdef _DEBUG
	printf("empty d_fetchVertex\n");
#endif
}

template<>
__forceinline__ __device__ void d_fetchVertex<CT_Rational>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	w = *i_w;
	v0 = i_vertex[0];
	v1 = i_vertex[1] * w;
	v2 = i_vertex[2];
}

template<>
__forceinline__ __device__ void d_fetchVertex<CT_Linear>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	v0 = i_vertex[0];
	v1 = i_vertex[1];
}

template<>
__forceinline__ __device__ void d_fetchVertex<CT_Quadratic>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	v0 = i_vertex[0];
	v1 = i_vertex[1];
	v2 = i_vertex[2];
}

template<>
__forceinline__ __device__ void d_fetchVertex<CT_Cubic>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	v0 = i_vertex[0];
	v1 = i_vertex[1];
	v2 = i_vertex[2];
	v3 = i_vertex[3];
}

__forceinline__ __device__ void d_fetchVertex(
	uint8_t curve_type,	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {

	if (curve_type == CT_Linear) {
		d_fetchVertex<CT_Linear>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else if (curve_type == CT_Quadratic) {
		d_fetchVertex<CT_Quadratic>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else if (curve_type == CT_Cubic) {
		d_fetchVertex<CT_Cubic>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else { //if (curve_type == CT_Rational) {
		d_fetchVertex<CT_Rational>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------


// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_storeVertex(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float w) {
#ifdef _DEBUG
	printf("empty d_storeVertex\n");
#endif
}

template<>
__forceinline__ __device__ void d_storeVertex<CT_Rational>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float w
	) {
	*i_w = w;
	i_vertex[0] = v0;
	i_vertex[1] = v1 / w;
	i_vertex[2] = v2;
}

template<>
__forceinline__ __device__ void d_storeVertex<CT_Linear>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float w
	) {
	i_vertex[0] = v0;
	i_vertex[1] = v1;
}

template<>
__forceinline__ __device__ void d_storeVertex<CT_Quadratic>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float w
	) {
	i_vertex[0] = v0;
	i_vertex[1] = v1;
	i_vertex[2] = v2;
}

template<>
__forceinline__ __device__ void d_storeVertex<CT_Cubic>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float w
	) {
	i_vertex[0] = v0;
	i_vertex[1] = v1;
	i_vertex[2] = v2;
	i_vertex[3] = v3;
}

__forceinline__ __device__ void d_storeVertex(
	uint8_t curve_type, float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float w
	) {

	if (curve_type == CT_Linear) {
		d_storeVertex<CT_Linear>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else if (curve_type == CT_Quadratic) {
		d_storeVertex<CT_Quadratic>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else if (curve_type == CT_Cubic) {
		d_storeVertex<CT_Cubic>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else { //if (curve_type == CT_Rational) {
		d_storeVertex<CT_Rational>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_storeVertex_rev(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w) {
#ifdef _DEBUG
	printf("empty d_storeVertex\n");
#endif
}

template<>
__forceinline__ __device__ void d_storeVertex_rev<CT_Rational>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	w = *i_w;
	i_vertex[2] = v0;
	i_vertex[1] = v1 / w;
	i_vertex[0] = v2;
}

template<>
__forceinline__ __device__ void d_storeVertex_rev<CT_Linear>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	i_vertex[1] = v0;
	i_vertex[0] = v1;
}

template<>
__forceinline__ __device__ void d_storeVertex_rev<CT_Quadratic>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	i_vertex[2] = v0;
	i_vertex[1] = v1;
	i_vertex[0] = v2;
}

template<>
__forceinline__ __device__ void d_storeVertex_rev<CT_Cubic>(
	float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {
	i_vertex[3] = v0;
	i_vertex[2] = v1;
	i_vertex[1] = v2;
	i_vertex[0] = v3;
}

__forceinline__ __device__ void d_storeVertex_rev(
	uint8_t curve_type, float2 *i_vertex, float *i_w,
	float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w
	) {

	if (curve_type == CT_Linear) {
		d_storeVertex_rev<CT_Linear>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else if (curve_type == CT_Quadratic) {
		d_storeVertex_rev<CT_Quadratic>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else if (curve_type == CT_Cubic) {
		d_storeVertex_rev<CT_Cubic>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
	else { //if (curve_type == CT_Rational) {
		d_storeVertex_rev<CT_Rational>(i_vertex, i_w, v0, v1, v2, v3, w);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_fetch_vertex_misaligned(
	float2 *i_vertex, float *i_w, float2 &v0, float2&v1, float2 &v2, float2 &v3, float &w) {
	d_fetchVertex<CURVE_TYPE>(i_vertex, i_w, v0, v1, v2, v3, w);
	d_misalign<CURVE_TYPE>(v0, v1, v2, v3);
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ void d_fetchVertexToShared(
	float2 *i_vertex, float *i_w, uint32_t index, float *o_coord) {
#ifdef _DEBUG
	printf("empty d_fetchVertexToShared\n");
#endif
}

template<>
__forceinline__ __device__ void d_fetchVertexToShared<CT_Rational>(
	float2 *i_vertex, float *i_w, uint32_t index, float *o_coord) {
}

template<>
__forceinline__ __device__ void d_fetchVertexToShared<CT_Linear>(
	float2 *i_vertex, float *i_w, uint32_t index, float *o_coord) {
}

template<>
__forceinline__ __device__ void d_fetchVertexToShared<CT_Quadratic>(
	float2 *i_vertex, float *i_w, uint32_t index, float *o_coord) {
}

template<>
__forceinline__ __device__ void d_fetchVertexToShared<CT_Cubic>(
	float2 *i_vertex, float *i_w, uint32_t index, float *o_coord) {
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
template <int BIG_FRAG_SIZE>
__forceinline__ __device__ float2 d_last_vertex(float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
#ifdef _DEBUG
	printf("empty d_last_vertex\n");
#endif
	return make_float2(0.f, 0.f);
}

template <>
__forceinline__ __device__ float2 d_last_vertex<CT_Linear>(
	float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	return v1;
}

template <>
__forceinline__ __device__ float2 d_last_vertex<CT_Quadratic>(
	float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	return v2;
}

template <>
__forceinline__ __device__ float2 d_last_vertex<CT_Cubic>(
	float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	return v3;
}

template <>
__forceinline__ __device__ float2 d_last_vertex<CT_Rational>(
	float2 &v0, float2 &v1, float2 &v2, float2 &v3) {
	return v2;
}

} // end of namespace Rasterizers

} // end of namespace Mochimazui

#endif