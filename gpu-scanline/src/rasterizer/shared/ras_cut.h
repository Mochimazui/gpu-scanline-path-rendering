
#ifndef _MOCHIMAZUI_RASTERIZER_SHARED_CUT_H_
#define _MOCHIMAZUI_RASTERIZER_SHARED_CUT_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../cutil_math.h"
#include "ras_define.h"
#include "ras_device_func.h"

namespace Mochimazui {

namespace Rasterizer_R_Cut {

using namespace Rasterizer;

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ float solveLineEquation(float a, float b) {
	return -b / a;
}

__forceinline__ __device__ float solveLineBezier(float A, float B, float C) {

	// A*(1-t) + B*t = C

	float a = B - A;
	float b = A;
	float t = (C - b) / a;
	return t;
}

__forceinline__ __device__ void solveQuadEquation(float a, float b, float c, float *r0, float *r1) {
	if (a == 0) {
		float x = -c / b;
		*r0 = x;
		*r1 = x;
		return;
	}

	float A = a;
	float B = b * 0.5;
	float C = c;
	float2 tc;

	float R = B*B - A*C;
	if (R > 0.0f) {
		float SR = sqrtf(R);
		if (B > 0.0f) {
			float TB = B + SR;
			tc = make_float2(-C / TB, -TB / A);
		}
		else { // B<0
			float TB = -B + SR;
			tc = make_float2(TB / A, C / TB);
		}
	}
	else {
		tc = make_float2(0.f, 0.f);
	}

	*r0 = tc.x;
	*r1 = tc.y;
}

__forceinline__ __device__ float solveQuadBezier(float iA, float iB, float iC, float iD) {

	// A*(1-t)^2 + 2*B*t*(1-t) + C*t^2 = D

	float a = iA - 2.f * iB + iC;
	float b = -2.f * iA + 2.f * iB;
	float c = iA - iD;

	float r0, r1;
	solveQuadEquation(a, b, c, &r0, &r1);
	if (0.f <= r0 && r0 <= 1.f) { return r0; }
	if (0.f <= r1 && r1 <= 1.f) { return r1; }

	ASSERT(0);
	return 0.f;
}

__forceinline__ __device__ float solveBezQuadNormalized(const float y1, const float h)
{
	const float e = y1 - .5f;
	if (-.0125f < e && e < .0125f)
	{
		float t = h*(1.f + 2.f * e*(-1.f + h)*(1.f + e*(-2.f + 4.f * h))); // |e| < .0125
		return t;
	}
	else
	{
		return (y1 - sqrt(h - 2.f * h*y1 + y1*y1)) / (2.f * e);
	}
}


__forceinline__ __device__ float solveCubicBezierAnalytical(float A, float B, float C, float D, float E, float t0, float t1) {

	float a = -A + 3 * B - 3 * C + D;
	float b = 3 * A - 6 * B + 3 * C;
	float c = -3 * A + 3 * B;
	float d = A - E;

	//printf("%f\n", a);

	//if (a == 0) {
	if (fabsf(a) < 1.f / 65536.f) {
		float r0, r1;
		solveQuadEquation(b, c, d, &r0, &r1);
		if (t0 < r0 && r0 < t1) { return r0; }
		if (t0 < r1 && r1 < t1) { return r1; }
		return t0;
	}

	b /= 3.0f;
	c /= 3.0f;

	float s1 = a*c - b*b;
	float s2 = a*d - b*c;
	float s3 = b*d - c*c;

	float delta = 4 * s1 *s3 - s2*s2;

	if (delta < 0) {

		bool flag = b*b*b*d >= a*c*c*c;

		float aa, cc, dd;
		if (flag) {
			aa = a; cc = s1; dd = -2 * b * s1 + a * s2;
		}
		else {
			aa = d; cc = s3; dd = -d * s2 + 2 * c * s3;
		}

		float t0 = (dd < 0) ? -1 : 1;
		t0 = -t0 * abs(aa) * sqrt(-delta);
		float t1 = -dd + t0;
		float p = cbrtf(t1 / 2.f);

		float q;
		if (t0 == t1) { q = -p; }
		else { q = -cc / p; }

		float x1;
		if (cc <= 0) { x1 = p + q; }
		else { x1 = -dd / (p*p + q*q + cc); }

		if (flag) { x1 = (x1 - b) / a; }
		else { x1 = -d / (x1 + c); }

		if (t0 < x1 && x1 < t1) { return x1; }

		//printf("---- d<0 ----\nt0:%f t1:%f p:%f q:%f\n(%f %f %f %f) %f\n",
		//	t0, t1, p, q, a, b, c, d, x1);

		return t0;

		//return x1;

	}
	else {

		// root l
		//float aa = a;
		float ca = s1;
		float da = -2 * b * s1 + a * s2;

		float ta = abs(atan2(a*sqrt(delta), -da)) / 3.f;
		float x1a = 2 * sqrt(-ca) *cos(ta);
		float x3a = 2 * sqrt(-ca) *(-0.5f *cos(ta) - (sqrt(3.f) / 2.f) * sin(ta));
		float xl;
		if (x1a + x3a > 2 * b) { xl = x1a; }
		else { xl = x3a; }
		float wl = a;
		xl = (xl - b) / a;

		// root s
		//float ad = d;
		float cd = s3;
		float dd = -d * s2 + 2 * c *s3;

		float td = abs(atan2(d*sqrt(delta), -dd)) / 3.f;
		float x1d = 2 * sqrt(-cd) *cos(td);
		float x3d = 2 * sqrt(-cd) *(-0.5f *cos(td) - (sqrt(3.f) / 2.f) * sin(td));
		float xs;
		if (x1d + x3d < 2 * c) { xs = x1d; }
		else { xs = x3d; }
		float ws = xs + c;
		xs = -d / ws;

		// root 3
		float e = wl * ws;
		float f = -xl * ws - wl * xs;
		float g = xl * xs;
		float xm = (c*f - b*g) / (-b*f + c*e);

		//

		if (t0 < xl && xl < t1) { return xl; }
		if (t0 < xs && xs < t1) { return xs; }
		if (t0 < xm && xm < t1) { return xm; }

		return t0;
	}

}


__forceinline__ __device__ float solveCubicBezier(float A, float B, float C, float D, float E) {

#define ITERATION
#ifdef ITERATION
{
	float y1 = (B - A) / (D - A);
	float y2 = (C - A) / (D - A);
	float h = (E - A) / (D - A);

	float y1appx = (-1 + 3 * y1 + 3 * y2)*.25f;
	float t = solveBezQuadNormalized(y1appx, h);

	// use newton iterations. this should converge very quickly with a good guess
	float dt = 1;
	float delta3 = 3.f*(y1 - y2);

#pragma unroll
	for (int i = 0; i < 4; ++i) {
		dt = (-h + t*(3.f * y1 + t*(t - 6.f * y1 + delta3*t + 3.f * y2))) / (3.f*(y1 + t*(t - 4.f * y1 + delta3 * t + 2.f * y2)));
		t -= dt;
	}

	//while (dt > 1e-6 || dt < -1e-6)
	//{
	//	dt = (-h + t*(3 * y1 + t*(t - 6 * y1 + 3 * t*y1 + 3 * y2 - 3 * t*y2))) / (3.*(y1 + t*(t - 4 * y1 + 3 * t*y1 + 2 * y2 - 3 * t*y2)));
	//	t -= dt;
	//}

	return t;
}

#else

	float a = -A + 3 * B - 3 * C + D;
	float b = 3 * A - 6 * B + 3 * C;
	float c = -3 * A + 3 * B;
	float d = A - E;

	if (a == 0) {
		float r0, r1;
		solveQuadEquation(b, c, d, &r0, &r1);
		if (0.f <= r0 && r0 <= 1.f) { return r0; }
		if (0.f <= r1 && r1 <= 1.f) { return r1; }
		return 0.f;
	}

	b /= a;
	c /= a;
	d /= a;

	auto solve = [](float a, float b, float c,
		float *x0, float *x1, float *x2) {

		const float M_PI = 3.1415926535;

		float q = (a * a - 3 * b);
		float r = (2 * a * a * a - 9 * a * b + 27 * c);

		float Q = q / 9;
		float R = r / 54;

		float Q3 = Q * Q * Q;
		float R2 = R * R;

		float CR2 = 729 * r * r;
		float CQ3 = 2916 * q * q * q;

		if (R == 0 && Q == 0)
		{
			*x0 = -a / 3;
			*x1 = -a / 3;
			*x2 = -a / 3;
			return 3;
		}
		else if (CR2 == CQ3)
		{
			/* this test is actually R2 == Q3, written in a form suitable
			for exact computation with integers */

			/* Due to finite precision some float roots may be missed, and
			considered to be a pair of complex roots z = x +/- epsilon i
			close to the real axis. */

			float sqrtQ = sqrt(Q);

			if (R > 0)
			{
				*x0 = -2 * sqrtQ - a / 3;
				*x1 = sqrtQ - a / 3;
				*x2 = sqrtQ - a / 3;
			}
			else
			{
				*x0 = -sqrtQ - a / 3;
				*x1 = -sqrtQ - a / 3;
				*x2 = 2 * sqrtQ - a / 3;
			}
			return 3;
		}
		else if (R2 < Q3)
		{
			float sgnR = (R >= 0 ? 1 : -1);
			float ratio = sgnR * sqrt(R2 / Q3);
			float theta = acos(ratio);
			float norm = -2 * sqrt(Q);
			*x0 = norm * cos(theta / 3) - a / 3;
			*x1 = norm * cos((theta + 2.0 * M_PI) / 3) - a / 3;
			*x2 = norm * cos((theta - 2.0 * M_PI) / 3) - a / 3;

			/* Sort *x0, *x1, *x2 into increasing order */

			if (*x0 > *x1) {
				auto temp = *x0;
				*x0 = *x1;
				*x1 = temp;
			}

			if (*x1 > *x2)
			{
				{
					auto temp = *x1;
					*x1 = *x2;
					*x2 = temp;
				}

				if (*x0 > *x1) {
					auto temp = *x0;
					*x0 = *x1;
					*x1 = temp;
				}
			}

			return 3;
		}
		else
		{
			float sgnR = (R >= 0 ? 1 : -1);
			float A = -sgnR * pow(abs(R) + sqrt(R2 - Q3), 1.f / 3.f);
			float B = Q / A;
			*x0 = A + B - a / 3;
			return 1;
		}

	};

	float r0, r1, r2;

	solve(b, c, d, &r0, &r1, &r2);

	if (0.f <= r0 && r0 <= 1.f) { return r0; }
	if (0.f <= r1 && r1 <= 1.f) { return r1; }
	if (0.f <= r2 && r2 <= 1.f) { return r2; }

	return 0.f;

#endif
}

// -------- -------- -------- -------- -------- -------- -------- --------

//go for the original approach and compute t here: it's not THAT costly, and avoids saving an int4 array
//template<int ORDER>
__forceinline__ __device__ float solveQuadBezierQM(float iA, float iB, float iC, float iD, float t0, float t1) {

	// A*(1-t)^2 + 2*B*t*(1-t) + C*t^2 = D

	float a = iA + (-2.f) * iB + iC;
	float b = -2.f * iA + 2.f * iB;
	float c = iA - iD;

	float r0 = 0.f, r1 = 0.f;
	solveQuadEquation(a, b, c, &r0, &r1);
	if (t0 <= r0 && r0 <= t1) { return r0; }
	if (t0 <= r1 && r1 <= t1) { return r1; }

	ASSERT(0);
	return t0;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ float cubic_interpolate(float v0, float v1, float v2, float v3, float t) {
	float qx0 = (1 - t) * v0 + t * v1;
	float qx1 = (1 - t) * v1 + t * v2;
	float qx2 = (1 - t) * v2 + t * v3;

	float lx0 = (1 - t) * qx0 + t * qx1;
	float lx1 = (1 - t) * qx1 + t * qx2;

	return (1 - t) * lx0 + t *lx1;
};

// -------- -------- -------- -------- -------- -------- -------- --------
__forceinline__ __device__ float solveCubicBezierQM(float v0, float v1, float v2, float v3, float v, float t0) {
	float t1 = 1.f;
	float vt0 = cubic_interpolate(v0, v1, v2, v3, t0);
	float vt1 = cubic_interpolate(v0, v1, v2, v3, t1);
	float t_ret = t0;
#pragma unroll
	for (int i = 0; i < 4; ++i) {
		//#define CUBIC_USE_QUADRATIC_ITERATION
		float th = (t0 + t1)*0.5f;
		float vth = cubic_interpolate(v0, v1, v2, v3, th);
		if ((__float_as_int(vth - v) ^ __float_as_int(vt0 - v)) >= 0) {
			t0 = th;
			vt0 = vth;
		}
		else {
			t1 = th;
			vt1 = vth;
		}
		float tm = (v - vt0)*safeRcp(vt1 - vt0)*(t1 - t0) + t0;
		/*
		#ifdef CUBIC_USE_QUADRATIC_ITERATION
		//quadratic
		float th=(t0+t1)*0.5f;
		float vth=cubic_interpolate(v0,v1,v2,v3, th);
		float ph=vth-vt0;
		float p1=vt1-vt0;
		float a=2.f*p1+(-4.f)*ph;
		float b=(4.f*ph-p1);
		float c=vt0-v;
		////////////
		float r0=0.f,r1=0.f;
		solveQuadEquation(a,b,c, &r0,&r1);
		if(r0>0.f&&r0<1.f){tm=t0+(t1-t0)*r0;}
		else if(r1>0.f&&r1<1.f){tm=t0+(t1-t0)*r1;}
		////////////
		//if(GET_ID()==26522&&v==13.f){
		//printf("%f %f %f %f\n",r0,r1,tm,(v-vt0)*safeRcp(vt1-vt0)*(t1-t0)+t0);
		//}
		#endif
		*/
		float vtm = cubic_interpolate(v0, v1, v2, v3, tm);
		//if(GET_ID()==26522&&v==13.f){
		//printf("%d %d  %f %f %f  %f %f %f  %f\n",GET_ID(), i,t0,tm,t1, vt0,vtm,vt1, v);
		//}
		//same sign test
		if ((__float_as_int(vtm - v) ^ __float_as_int(vt0 - v)) >= 0) {
			t0 = tm;
			vt0 = vtm;
		}
		else {
			t1 = tm;
			vt1 = vtm;
		}
		t_ret = tm;
	}
	return t_ret;
}

#define MODE_LINEAR CT_Linear
#define MODE_QUADRATIC CT_Quadratic
#define MODE_CUBIC CT_Cubic
#define MODE_RATIONAL CT_Rational

////////////
#define T1_QUEUE(i) (t1_queue[(i)*BLOCK_SIZE])
#define POINT_COORDS(i) (point_coords[(i)*BLOCK_SIZE])

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE>
__forceinline__ __device__ float d_interpolate_general_curve(float v0, float v1, float v2, float v3, float t) {
#ifdef _DEBUG
	printf("empty d_interpolate_general_curve\n");
#endif
	return 0.f;
}

template<>
__forceinline__ __device__ float d_interpolate_general_curve<CT_Cubic>(
	float v0, float v1, float v2, float v3, float t) {

	float qx0 = (1 - t) * v0 + t * v1;
	float qx1 = (1 - t) * v1 + t * v2;
	float qx2 = (1 - t) * v2 + t * v3;

	float lx0 = (1 - t) * qx0 + t * qx1;
	float lx1 = (1 - t) * qx1 + t * qx2;

	return (1 - t) * lx0 + t *lx1;
}

// -------- -------- -------- -------- -------- -------- -------- --------

template<int BLOCK_SIZE, int mode>
__forceinline__ __device__ void d_make_intersection_shared_to_reg(
	float* point_coords,
	float &v0, float &v1, float &v2, float &v3) {

	if (mode == CT_Linear) {
		v0 = POINT_COORDS(0);
		v1 = POINT_COORDS(1);
	}
	else if (mode == CT_Quadratic) {
		v0 = POINT_COORDS(0);
		v1 = POINT_COORDS(1);
		v2 = POINT_COORDS(2);
	}
	else if (mode == CT_Cubic) {
		v0 = POINT_COORDS(0);
		v1 = POINT_COORDS(1);
		v2 = POINT_COORDS(2);
		v3 = POINT_COORDS(3);
	}
	else if (mode == CT_Rational) {
		v0 = POINT_COORDS(0);
		v1 = POINT_COORDS(1);
		v2 = POINT_COORDS(2);
	}
	else {
		ASSERT(0);
	}
}

template<int BLOCK_SIZE, int mode>
__forceinline__ __device__ float interpolateGeneralCurve(float arcw1, float* point_coords, float t) {
	if (mode == CT_Linear) {
		float v0 = POINT_COORDS(0);
		float v1 = POINT_COORDS(1);
		return ::lerp(v0, v1, t);

	}
	else if (mode == CT_Quadratic) {
		float v0 = POINT_COORDS(0);
		float v1 = POINT_COORDS(1);
		float v2 = POINT_COORDS(2);

		float lv0 = ::lerp(v0, v1, t);
		float lv1 = ::lerp(v1, v2, t);

		return ::lerp(lv0, lv1, t);
	}
	else if (mode == CT_Cubic) {
		float iv0 = POINT_COORDS(0);
		float iv1 = POINT_COORDS(1);
		float iv2 = POINT_COORDS(2);
		float iv3 = POINT_COORDS(3);

		float qv0 = ::lerp(iv0, iv1, t);
		float qv1 = ::lerp(iv1, iv2, t);
		float qv2 = ::lerp(iv2, iv3, t);

		float lv0 = ::lerp(qv0, qv1, t);
		float lv1 = ::lerp(qv1, qv2, t);

		return ::lerp(lv0, lv1, t);
	}
	else if (mode == CT_Rational) {
		float iv0 = POINT_COORDS(0);
		float iv1 = POINT_COORDS(1)*arcw1;
		float iv2 = POINT_COORDS(2);

		auto iw = arcw1;

		float lv0 = ::lerp(iv0, iv1, t);
		float lv1 = ::lerp(iv1, iv2, t);
		float v = ::lerp(lv0, lv1, t);
		float w = (1 - t)*(1 - t) + 2.f*(1 - t)*t*iw + t*t;
		v *= safeRcp(w);
		return v;

		//float lx0 = (1 - t) * v0 + t * v1;
		//float lx1 = (1 - t) * v1 + t * v2;

		//float v_up = (1 - t) * lx0 + t *lx1;
		//float v_down = (1 - t)*(1 - t) + 2.f*(1 - t)*t*arcw1 + t*t;

		//return v_up*safeRcp(v_down);
	}
	else {
		ASSERT(0);
		return 0.f;
	}
}

template<int BLOCK_SIZE, int mode>
__forceinline__ __device__ float interpolateGeneralCurve(
	float arcw1, float v0, float v1, float v2, float v3, float t) {

	if (mode == CT_Linear) {
		return ::lerp(v0, v1, t);
	}
	else if (mode == CT_Quadratic) {
		float lv0 = ::lerp(v0, v1, t);
		float lv1 = ::lerp(v1, v2, t);

		return ::lerp(lv0, lv1, t);
	}
	else if (mode == CT_Cubic) {
		float iv0 = v0;
		float iv1 = v1;
		float iv2 = v2;
		float iv3 = v3;

		float qv0 = ::lerp(iv0, iv1, t);
		float qv1 = ::lerp(iv1, iv2, t);
		float qv2 = ::lerp(iv2, iv3, t);

		float lv0 = ::lerp(qv0, qv1, t);
		float lv1 = ::lerp(qv1, qv2, t);

		return ::lerp(lv0, lv1, t);
	}
	else if (mode == CT_Rational) {
		float iv0 = v0;
		float iv1 = v1 *arcw1;
		float iv2 = v2;

		auto iw = arcw1;

		float lv0 = ::lerp(iv0, iv1, t);
		float lv1 = ::lerp(iv1, iv2, t);
		float v = ::lerp(lv0, lv1, t);
		float w = (1 - t)*(1 - t) + 2.f*(1 - t)*t*iw + t*t;
		v *= safeRcp(w);
		return v;
	}
	else {
		ASSERT(0);
		return 0.f;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

template<int D>
__forceinline__ __device__ int d_float2int_rd(float x) {
	return __float2int_rd(x / D) * D;
}

template<>
__forceinline__ __device__ int d_float2int_rd<1>(float x) {
	return __float2int_rd(x);
}

template<>
__forceinline__ __device__ int d_float2int_rd<2>(float x) {
	return __float2int_rd(x) & 0xFFFFFFFE;
}

template<>
__forceinline__ __device__ int d_float2int_rd<4>(float x) {
	return __float2int_rd(x) & 0xFFFFFFFC;
}

template<int D>
__forceinline__ __device__ int d_round_down(int x) {
	return x;
}

template<>
__forceinline__ __device__ int d_round_down<2>(int x) {
	return x & 0xFFFFFFFE;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int FRAG_SIZE>
__forceinline__ __device__ void d_get_xy_begin_end(
	float2 v0, float2 v1, int &xbegin, int &xend, int &ybegin, int &yend) {

	if (v0.x <= v1.x) {
		xbegin = d_float2int_rd<FRAG_SIZE>(v0.x) + FRAG_SIZE;
		xend = d_float2int_rd<FRAG_SIZE>(v1.x);
	}
	else {
		xbegin = d_float2int_rd<FRAG_SIZE>(v1.x) + FRAG_SIZE;
		xend = d_float2int_rd<FRAG_SIZE>(v0.x);
	}
	if (v0.y <= v1.y) {
		ybegin = d_float2int_rd<FRAG_SIZE>(v0.y) + FRAG_SIZE;
		yend = d_float2int_rd<FRAG_SIZE>(v1.y);
	}
	else {
		ybegin = d_float2int_rd<FRAG_SIZE>(v1.y) + FRAG_SIZE;
		yend = d_float2int_rd<FRAG_SIZE>(v0.y);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int FRAG_SIZE>
__forceinline__ __device__ void d_get_xy_begin_end_delta(
	float2 v0, float2 v1, int &xbegin, int &xend, int &ybegin, int &yend, float &dx, float &dy) {

	if (v0.x <= v1.x) {
		xbegin = d_float2int_rd<FRAG_SIZE>(v0.x) + FRAG_SIZE;
		xend = d_float2int_rd<FRAG_SIZE>(v1.x);
		dx = FRAG_SIZE;
	}
	else {
		xbegin = d_float2int_rd<FRAG_SIZE>(v1.x) + FRAG_SIZE;
		xend = d_float2int_rd<FRAG_SIZE>(v0.x);
		dx = -FRAG_SIZE;
	}
	if (v0.y <= v1.y) {
		ybegin = d_float2int_rd<FRAG_SIZE>(v0.y) + FRAG_SIZE;
		yend = d_float2int_rd<FRAG_SIZE>(v1.y);
		dy = FRAG_SIZE;
	}
	else {
		ybegin = d_float2int_rd<FRAG_SIZE>(v1.y) + FRAG_SIZE;
		yend = d_float2int_rd<FRAG_SIZE>(v0.y);
		dy = -FRAG_SIZE;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int FRAG_SIZE>
__forceinline__ __device__ void d_udpate_cut_range(
	int width, int height,
	int &curve_x_begin, int &curve_x_end,
	int &curve_y_begin, int &curve_y_end,
	int &cut_n_x, int &cut_n_y
	) {

	int cut_x_min = 0;
	int cut_x_max = d_round_down<FRAG_SIZE>(width) + FRAG_SIZE;
	int cut_y_min = 0;
	int cut_y_max = d_round_down<FRAG_SIZE>(height) + FRAG_SIZE;

	if (
		(curve_x_begin < cut_x_min && curve_x_end < cut_x_min) 
		|| 
		(curve_x_begin > cut_x_max && curve_x_end > cut_x_max)
		|| 
		(curve_x_begin > curve_x_end)
		) {
		cut_n_x = 0;
	}
	else {
		curve_x_begin = limit(curve_x_begin, cut_x_min, cut_x_max);
		curve_x_end = limit(curve_x_end, cut_x_min, cut_x_max);

		cut_n_x = max((curve_x_end - curve_x_begin) / FRAG_SIZE + 1, 0);
	}

	if (
		(curve_y_begin < cut_y_min && curve_y_end < cut_y_min)
		||
		(curve_y_begin > cut_y_max && curve_y_end > cut_y_max)
		||
		(curve_y_begin > curve_y_end)
		) {
		cut_n_y = 0;
	}
	else {
		curve_y_begin = limit(curve_y_begin, cut_y_min, cut_y_max);
		curve_y_end = limit(curve_y_end, cut_y_min, cut_y_max);

		cut_n_y = max((curve_y_end - curve_y_begin) / FRAG_SIZE + 1, 0);
	}

	//curve_x_begin = max(curve_x_begin, -FRAG_SIZE);
	//curve_x_end = min(curve_x_end, d_round_down<FRAG_SIZE>(width) + FRAG_SIZE);
	//curve_y_begin = max(curve_y_begin, -FRAG_SIZE);
	//curve_y_end = min(curve_y_end, d_round_down<FRAG_SIZE>(height) + FRAG_SIZE);

	//curve_x_begin = limit(curve_x_begin, cut_x_min, cut_x_max);
	//curve_x_end = limit(curve_x_end, cut_x_min, cut_x_max);
	//curve_y_begin = limit(curve_y_begin, cut_y_min, cut_y_max);
	//curve_y_end = limit(curve_y_end, cut_y_min, cut_y_max);

	//cut_n_x = max((curve_x_end - curve_x_begin) / FRAG_SIZE + 1, 0);
	//cut_n_y = max((curve_y_end - curve_y_begin) / FRAG_SIZE + 1, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE, int FRAG_SIZE, int BLOCK_SIZE>
__forceinline__ __device__ void k_make_intersection_impl_0(
	int cidx_i0, float arcw1,
	float* t1_queue, int n_cuts, float* point_coords,
	int2* ts, int* pcnt, int n_fragments, int w, int h) {

	//int i0 = GET_ID();
	//float t0_ms = 0.f;
	float2 v0_ms = make_float2(POINT_COORDS(0), POINT_COORDS(4));

	int p = 0;

	for (int ms_i = 0; ms_i<n_cuts; ms_i++) {
		float t1_ms = T1_QUEUE(ms_i);
		float2 v1_ms = make_float2(
			interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(arcw1, point_coords, t1_ms),
			interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(arcw1, point_coords + 4 * BLOCK_SIZE, t1_ms)
			);

		// -------- --------
		int curve_x_begin, curve_x_end;
		int curve_y_begin, curve_y_end;

		d_get_xy_begin_end<FRAG_SIZE>(v0_ms, v1_ms, 
			curve_x_begin, curve_x_end, curve_y_begin, curve_y_end);

		int cut_n_x = 0;
		int cut_n_y = 0;

		d_udpate_cut_range<FRAG_SIZE>(
			w, h,
			curve_x_begin, curve_x_end,
			curve_y_begin, curve_y_end,
			cut_n_x, cut_n_y
			);

		//we need to keep x-

		//int xbegin, xend, ybegin, yend;
		//xbegin = curve_x_begin;
		//xend = curve_x_end;
		//ybegin = curve_y_begin;
		//yend = curve_y_end;

		int n_x = cut_n_x;
		int n_y = cut_n_y;

		p += 1 + n_x + n_y;

		//t0_ms = t1_ms;
		v0_ms = v1_ms;
	}

	pcnt[cidx_i0] = p;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<int CURVE_TYPE, int FRAG_SIZE, int BLOCK_SIZE>
__forceinline__ __device__ void k_make_intersection_impl_1(
	int cidx_i0, float arcw1,
	float* t1_queue, int n_cuts, float* point_coords,
	int2* ts, int* pcnt, int n_fragments, int w, int h) {

//#define PRINT_DEBUG_INFO

	float t0_ms = 0.f;

	float2 v0_ms = make_float2(POINT_COORDS(0), POINT_COORDS(4));
	int p = 0;

	p = pcnt[cidx_i0];

	for (int ms_i = 0; ms_i < n_cuts; ms_i++) {
		float t1_ms = T1_QUEUE(ms_i);
		float2 v1_ms = make_float2(
			interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(arcw1, point_coords, t1_ms),
			interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(arcw1, point_coords + 4 * BLOCK_SIZE, t1_ms)
			);
		t1_ms = __int_as_float((__float_as_int(t1_ms) & 0xFFFFFFFC));

		if (floor(v1_ms.x) == v1_ms.x) {
			t1_ms = __int_as_float((__float_as_int(t1_ms) & 0xFFFFFFFC) | 2);
		}
		else {
			t1_ms = __int_as_float(__float_as_int(t1_ms) | 3);
		}

		// -------- --------
		int curve_x_begin, curve_x_end;
		int curve_y_begin, curve_y_end;

		float dx, dy;

		d_get_xy_begin_end_delta<FRAG_SIZE>(v0_ms, v1_ms,
			curve_x_begin, curve_x_end, curve_y_begin, curve_y_end,
			dx, dy);

		int cut_n_x = 0;
		int cut_n_y = 0;

#ifdef PRINT_DEBUG_INFO
		printf("curve-raw: %d %d/%d (%f %f) (%d %d - %d) (%d %d - %d)\n",
			cidx_i0, ms_i + 1, n_cuts, t0_ms, t1_ms,
			curve_x_begin, curve_x_end, cut_n_x,
			curve_y_begin, curve_y_end, cut_n_y);
#endif

		d_udpate_cut_range<FRAG_SIZE>(
			w, h,
			curve_x_begin, curve_x_end,
			curve_y_begin, curve_y_end,
			cut_n_x, cut_n_y
			);

		int xbegin, xend, ybegin, yend;

		xbegin = curve_x_begin;
		xend = curve_x_end;
		ybegin = curve_y_begin;
		yend = curve_y_end;

#ifdef PRINT_DEBUG_INFO
		printf("curve-cut: %d %d/%d (%f %f) (%d %d - %d) (%d %d - %d)\n"
			"\t(%f %f)(%f %f)\n" ,
			cidx_i0, ms_i +1, n_cuts, t0_ms, t1_ms,
			xbegin, xend, cut_n_x,
			ybegin, yend, cut_n_y,
			v0_ms.x, v0_ms.y, v1_ms.x, v1_ms.y
			);
#endif

		int n_x = cut_n_x;
		int n_y = cut_n_y;

		int n_loop = n_x + n_y + 1;

		//if (v0_ms.x >= w && v1_ms.x >= w) { n_x = 0; }

		float x = (float)(dx < 0.f ? xend : xbegin);
		float y = (float)(dy < 0.f ? yend : ybegin);

		//if (i0 == 0) { printf("BE: %d %d %d %d\n", xbegin, xend, ybegin, yend); }
		//if (i0 == 0) { printf("XY: %f %f\n", x, y); }

		float4 knots = make_float4(0.f, 0.f, 0.f, 0.f);

		//POINT_COORDS(8) = n_x ? t0_ms : 2.f; //tx
		//POINT_COORDS(9) = n_y ? t0_ms : 2.f; //ty

		POINT_COORDS(8) = t0_ms; //tx
		POINT_COORDS(9) = t0_ms; //ty

		int i_ts_last = __float_as_int(-1);

#ifdef PRINT_DEBUG_INFO
		float last_cut_c = -1.f;
#endif
		// start from -1, evaluate one 
		for (int i = -1; i < n_loop; i++) {
			float t_solve = 0.f;
			float c = 0.f;
			int side = 0;

			// get current t 
			auto next_tx = POINT_COORDS(8);
			auto next_ty = POINT_COORDS(9);

			float t_min;

			if (i == -1) { 
				// next_tx == next_ty == t0_ms
				t_min = next_tx;
				if (n_x == 0) {
					side = 0;
					t_solve = 2.f;
				}
				else if (n_y == 0) {
					side = 1;
					t_solve = 2.f;
				}
				else {
					side = 0;
					n_x--;
					c = x; x += dx;
				}
			}
			else if (next_tx <= next_ty) {
				t_min = next_tx;
				side = 0;
				if (n_x > 0) {
					n_x--;
					c = x; x += dx;
				}
				else {
					t_solve = 2.f;
				}
			}
			else {
				t_min = next_ty;
				side = 1;
				if (n_y > 0) {
					n_y--;
					c = y; y += dy;
				}
				else {
					t_solve = 2.f;
				}
			}

			//float t_min = POINT_COORDS(8 + side);

			if (i >= 0) {

				float f_ts_out = t_min;
				int i_ts_out = __float_as_int(f_ts_out);

				int i_t_out = __float_as_int(f_ts_out) & 0xFFFFFFFC;
				if (i_t_out == (i_ts_last & 0xFFFFFFFC)) {
					i_ts_out = i_ts_out | i_ts_last;
					ts[p - 1] = make_int2(cidx_i0, i_ts_out);
				}

				//printf("MI: %d %f %d\n", p, t_min, i_ts_out & 1); 

#ifdef PRINT_DEBUG_INFO
				printf("%d %d/%d %d/%d %d (%f %f) %d %f %f\n",
					cidx_i0, ms_i + 1, n_cuts, i, n_loop,
					p, next_tx, next_ty,
					side, t_min, last_cut_c
					);
#endif

				ts[p] = make_int2(cidx_i0, i_ts_out);
				i_ts_last = i_ts_out;
				p++;
			}

#ifdef PRINT_DEBUG_INFO
			last_cut_c = c;
#endif
			if (t_solve < 2.f) {

				if (CURVE_TYPE == MODE_LINEAR) {
					float y0 = POINT_COORDS(side * 4 + 0);
					float y1 = POINT_COORDS(side * 4 + 1);
					t_solve = min(max((c - y0)*safeRcp(y1 - y0), t_min), t1_ms);
				}
				else if (CURVE_TYPE == MODE_QUADRATIC) {
					float y0 = POINT_COORDS(side * 4 + 0);
					float y1 = POINT_COORDS(side * 4 + 1);
					float y2 = POINT_COORDS(side * 4 + 2);
					t_solve = solveQuadBezierQM(y0, y1, y2, c, t_min, t1_ms);
				}
				else if (CURVE_TYPE == MODE_RATIONAL) {
					float y0 = POINT_COORDS(side * 4 + 0);
					float y1 = POINT_COORDS(side * 4 + 1);
					float y2 = POINT_COORDS(side * 4 + 2);
					t_solve = solveQuadBezierQM(y0 - c, (y1 - c)*arcw1, y2 - c, 0.f, t_min, t1_ms);
				}
				else {

					// shared to reg
					float cv0, cv1, cv2, cv3;
					d_make_intersection_shared_to_reg<BLOCK_SIZE, CURVE_TYPE>(
						point_coords + side*(4 * BLOCK_SIZE), cv0, cv1, cv2, cv3);

					//
					float t0 = t_min;
					float t1 = t1_ms;
					float vt0 = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
						arcw1, cv0, cv1, cv2, cv3, t0);
					//float vt1 = (side ? v1_ms.y : v1_ms.x);
					t_solve = t0;

					// 
					if (vt0 != c) {

#define LR_CUBIC_BISECTION
#ifdef LR_CUBIC_BISECTION

						auto raw_t0 = t0;
						//auto raw_t1 = t1;
						//auto raw_vt0 = vt0;
						auto last_vtm = 0.f;

#define CUBIC_ITERATION_NUMBER 24
						//float left[CUBIC_ITERATION_NUMBER];
						//float right[CUBIC_ITERATION_NUMBER];
						//float mid[CUBIC_ITERATION_NUMBER];
#pragma unroll
						for (int jiter = 0; jiter < CUBIC_ITERATION_NUMBER; ++jiter) {
						//for (int jiter = 0; jiter < 64; ++jiter) {

							auto tm = (t0 + t1) * .5f;
							float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, cv0, cv1, cv2, cv3, tm);

							t_solve = tm;
							last_vtm = vtm;
							//if (vtm == c) { break; }
							//if (abs(vtm - c) < 0.0001) { break; }
							//if (t0 == t1) { break; }

							if ((__float_as_int(vtm - c) ^ __float_as_int(vt0 - c)) >= 0) {
								t0 = tm;
								vt0 = vtm;
							}
							else {
								t1 = tm;
								//vt1 = vtm;
							}

							//if (jiter == (CUBIC_ITERATION_NUMBER - 1)) {
							//	printf("%f\n", abs(vtm - c));
							//}

						}

						if (abs(last_vtm - c) > 1.f) {

							t_solve = raw_t0;
							auto tm = raw_t0;
							float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, cv0, cv1, cv2, cv3, tm);

							//printf("%f\n", abs(vtm - c));
						}
#endif

//#define LR_CUBIC_ITERATION
#ifdef LR_CUBIC_ITERATION
#define CUBIC_ITERATION_NUMBER 16

						auto last_vtm = 0.f;
						auto raw_t0 = t0;

#pragma unroll
						for (int jiter = 0; jiter < CUBIC_ITERATION_NUMBER; ++jiter) {

							float tm = (c - vt0)*safeRcp(vt1 - vt0)*(t1 - t0) + t0;
							float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, point_coords + side*(4 * BLOCK_SIZE), tm);
							float th = tm;

							t_solve = tm;
							if (vtm == c) { break; }
							if (abs(vtm - c) < 0.0001) { break; }

							if (jiter == (CUBIC_ITERATION_NUMBER - 1)) {
								printf("%f\n", abs(vtm - c));
							}

							if ((__float_as_int(vtm - c) ^ __float_as_int(vt0 - c)) >= 0) {
								t0 = tm;
								vt0 = vtm;
							}
							else {
								t1 = tm;
								vt1 = vtm;
							}
						}

						if (abs(last_vtm - c) > 0.01f) {

							t_solve = raw_t0;
							auto tm = raw_t0;
							float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, cv0, cv1, cv2, cv3, tm);

							printf("%f\n", abs(vtm - c));
						}

#endif

//#define QM_CUBIC_ITERATION
#ifdef QM_CUBIC_ITERATION

#pragma unroll

						auto last_vtm = 0.f;
						auto raw_t0 = t0;

						for (int jiter = 0; jiter < 6; ++jiter) {

							float tm = (c - vt0)*safeRcp(vt1 - vt0)*(t1 - t0) + t0;
							//float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
							//	arcw1, point_coords + side*(4 * BLOCK_SIZE), tm);
							float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, cv0, cv1, cv2, cv3, tm);
							float th = tm;

							//if (cidx_i0 == 3 && ms_i == 2 && c == 786) {
							//	printf("solve cubic: cid:%d sid:%d i:%d\n"
							//		"\t(%f %f) (%f %f)\n"
							//		"\t%f %f\n",
							//		cidx_i0, ms_i + 1, i,
							//		t0, t1, vt0, vt1,
							//		tm, vtm
							//		);
							//}

							//same sign test
							if ((__float_as_int(vtm - c) ^ __float_as_int(vt0 - c)) >= 0) {
								t0 = tm;
								th = t0 + (t1 - t0)*(1.f / 8.f);
								vt0 = vtm;
							}
							else {
								t1 = tm;
								th = t0 + (t1 - t0)*(7.f / 8.f);
								vt1 = vtm;
							}

							t_solve = tm;
							last_vtm = vtm;

							//float vth = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
							//	arcw1, point_coords + side*(4 * BLOCK_SIZE), th);
							float vth = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, cv0, cv1, cv2, cv3, th);
							if ((__float_as_int(vth - c) ^ __float_as_int(vt0 - c)) >= 0) {
								t0 = th;
								vt0 = vth;
							}
							else {
								t1 = th;
								vt1 = vth;
							}

						}

						if (abs(last_vtm - c) > 0.01f) {

							t_solve = raw_t0;
							auto tm = raw_t0;
							float vtm = interpolateGeneralCurve<BLOCK_SIZE, CURVE_TYPE>(
								arcw1, cv0, cv1, cv2, cv3, tm);

							//printf("%f\n", abs(vtm - c));
						}
#endif


					}

				}
			}

			//POINT_COORDS(8 + side) = t_solve;
			POINT_COORDS(8 + side) = __int_as_float((__float_as_int(t_solve) & 0xFFFFFFFC) | side);
		}

		t0_ms = t1_ms;
		v0_ms = v1_ms;
	}

	//if (!i0) {
	//	//tag for fragment generation
	//	ts[n_fragments] = make_int2(-1, 0x3f800000);
	//}

}

#define MAKE_CASE_MODE(CODE,MODE) \
		case MODE:{const int mode=MODE;CODE;break;}
#define TEMPLATIFY_MODES(CODE) \
		switch(mode){\
		default:{ASSERT(0);break;}\
		MAKE_CASE_MODE(CODE,MODE_LINEAR);\
		MAKE_CASE_MODE(CODE,MODE_QUADRATIC);\
		MAKE_CASE_MODE(CODE,MODE_CUBIC);\
		MAKE_CASE_MODE(CODE,MODE_RATIONAL);\
		}

template<int FRAG_SIZE, int BLOCK_SIZE>
__global__ void k_make_intersection_0(
	int2* ts, float4* t_cuts, char* n_cuts_cache, int* pcnt,

	uint32_t *i_curve_index,
	uint8_t *i_curve_type,
	uint32_t *i_curve_vertex_pos,

	int n_curves, int n_fragments,

	float2* vs, float* arcw1s, uint32_t *curve_path_id,

	int w, int h,
	int* cidx,
	uint64_t* is_path_visible
	) {

	//const int PASS = 0;

	int thread_index = GET_ID();
	if (thread_index >= n_curves) { return; }

	// ??? not necessary ???
	//int curve_index = i_curve_index[thread_index];
	int curve_index = thread_index;

	int i = thread_index;
	int i0 = i;
	int p0 = 0;
	int mode = 0;

	//we're able to deal with inflections - there are at most 4 cut points
	//do it in shared memory first - simpler
	//this uses up almost all the shared memory, hopefully it's worth it
	//POINT_COORDS(8) and POINT_COORDS(9) are temporary slots for the merge sort
	__shared__ float shared_t1_queue[5 * BLOCK_SIZE];
	__shared__ float shared_point_coords[10 * BLOCK_SIZE];

	//do it the spliced way to avoid bank conflicts
	float* t1_queue = shared_t1_queue + threadIdx.x;
	float* point_coords = shared_point_coords + threadIdx.x;
	float arcw1 = 0.f;

	p0 = i_curve_vertex_pos[curve_index];
	mode = i_curve_type[curve_index];
	if (mode == CT_Rational) { arcw1 = arcw1s[curve_index]; }

	//load the vertices
	//we need it when addressing anyway
	//float2* vsp0=vs+p0;
#pragma unroll
	for (int j = 0; j < 4; j++) {
		if (j < (mode & 7)) {
			float2 vi = vs[p0 + j];
			POINT_COORDS(j) = vi.x;
			POINT_COORDS(j + 4) = vi.y;
		}
	}

	//printf("C: %d (%f,%f)(%f,%f)\t\n(%f,%f) %f\n",
	//	i, vs[p0 + 0].x, vs[p0 + 0].y,
	//	vs[p0 + 1].x, vs[p0 + 1].y,
	//	vs[p0 + 2].x, vs[p0 + 2].y,
	//	arcw1);

	//compute the monotonic segments
	int n_cuts = 0;
	//bool is_visible = ((int)is_path_visible[curve_path_id[i0]] == 0x01010101);
	bool is_visible = PATH_VISIBLE(is_path_visible[curve_path_id[i0]]);

	// monotonize
	float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;

	if (is_visible) {
#pragma unroll
		for (int c = 0; c < 2; c++) {
			switch (mode) {
			default: {
				ASSERT(0);
				break;
			}case MODE_LINEAR: {
				//do nothing, it's monotone already
				break;
			}case MODE_QUADRATIC: {
				float y0 = POINT_COORDS(0 + c * 4);
				float y1 = POINT_COORDS(1 + c * 4);
				float y2 = POINT_COORDS(2 + c * 4);
				float b = (y0 - y1) + (y2 - y1);
				//they are in pixel units, absolute error should cut it
				if (fabsf(b) < 1e-6f) {
					// degenerate (is actually linear) so no min/max
				}
				else {
					// is actually quadratic, so solve
					float t = (y0 - y1) / b;
					if (t > 0.f&&t < 1.f) {
						T1_QUEUE(n_cuts) = t;
						n_cuts++;
					}
				}
				break;
			}case MODE_CUBIC: {
				float y0 = POINT_COORDS(0 + c * 4);
				float y1 = POINT_COORDS(1 + c * 4);
				float y2 = POINT_COORDS(2 + c * 4);
				float y3 = POINT_COORDS(3 + c * 4);

				//QM stablized version
				float r0 = 0.f, r1 = 0.f;
				solveQuadEquation(
					(y3 - y0) + 3.f*(y1 - y2),
					2.f*((y0 - y1) + (y2 - y1)),
					y1 - y0, &r0, &r1);

				if (r0 > 0.f&&r0 < 1.f) {
					T1_QUEUE(n_cuts) = r0;
					n_cuts++;
				}
				if (r1 > 0.f&&r1 < 1.f&&r1 != r0) {
					T1_QUEUE(n_cuts) = r1;
					n_cuts++;
				}

				break;
			}case MODE_RATIONAL: {
				float w0 = 1.f;
				float w1 = arcw1;
				float w2 = 1.f;
				float p0 = w0*POINT_COORDS(0 + c * 4);
				float p1 = w1*POINT_COORDS(1 + c * 4);
				float p2 = w2*POINT_COORDS(2 + c * 4);

				//float a = -2.f*p1*w0 + p2*w0 + 2.f*p0*w1 - p0*w2;
				//float d = sqrtf(p2*p2*w0*w0 + w2*(4.f*p1*p1*w0 - 4.f*p0*p1*w1 + p0*p0*w2) - 2.f*p2*(2.f*p1*w0*w1 - 2.f*p0*w1*w1 + p0*w0*w2));
				//float b = 2.f*(p2*(w0 - w1) + p0*(w1 - w2) + p1*(w2-w0));

				//QM hand-derived rewritten version: it's much more stable and has simpler code
				float pw01 = p0*w1 - p1*w0;
				float pw12 = p1*w2 - p2*w1;
				float pw20 = p2*w0 - p0*w2;

				float r0 = 0.f, r1 = 0.f;
				solveQuadEquation(
					pw01 + pw12 + pw20,
					-(2.f*pw01 + pw20),
					pw01, &r0, &r1);

				if (r0 > 0.f&&r0 < 1.f) {
					T1_QUEUE(n_cuts) = r0;
					n_cuts++;
				}
				if (r1 > 0.f&&r1 < 1.f&&r1 != r0) {
					T1_QUEUE(n_cuts) = r1;
					n_cuts++;
				}
				break;
			}
			}
		}

		//manual insertion sort
		q0 = T1_QUEUE(0); q1 = T1_QUEUE(1); q2 = T1_QUEUE(2); q3 = T1_QUEUE(3);

		if (n_cuts >= 2) {
			float t1 = (q1);
			float t0 = (q0);
			if (t1 < t0) {
				(q1) = t0;
				(q0) = t1;
			}
		}
		if (n_cuts >= 3) {
			float t2 = (q2);
			float t1 = (q1);
			if (t2 < t1) {
				(q2) = t1;
				float t0 = (q0);
				if (t2 < t0) {
					(q1) = t0;
					(q0) = t2;
				}
				else {
					(q1) = t2;
				}
			}
		}
		if (n_cuts >= 4) {
			float t3 = (q3);
			float t2 = (q2);
			if (t3 < t2) {
				(q3) = t2;
				float t1 = (q1);
				if (t3 < t1) {
					(q2) = t1;
					float t0 = (q0);
					if (t3 < t0) {
						(q1) = t0;
						(q0) = t3;
					}
					else {
						(q1) = t3;
					}
				}
				else {
					(q2) = t3;
				}
			}
		}
	}

	//cache in t_cuts
	T1_QUEUE(0) = q0;
	T1_QUEUE(1) = q1;
	T1_QUEUE(2) = q2;
	T1_QUEUE(3) = q3;
	t_cuts[i0] = make_float4(q0, q1, q2, q3);
	n_cuts_cache[i0] = n_cuts;

	if (is_visible) { T1_QUEUE(n_cuts) = 1.f; n_cuts++; }

	////////////////
	//loop over the segments and do the deed
	//int cidx_i0 = cidx[i0];
	int cidx_i0 = curve_index;
	TEMPLATIFY_MODES(
		(k_make_intersection_impl_0<mode, FRAG_SIZE, BLOCK_SIZE>(
			cidx_i0, arcw1, t1_queue, n_cuts, point_coords, ts, pcnt, n_fragments, w, h));
	);
}

template<int FRAG_SIZE, int BLOCK_SIZE>
__global__ void k_make_intersection_1(
	int2* ts, float4* t_cuts, char* n_cuts_cache, int* pcnt,

	uint32_t *i_curve_index,
	uint8_t *i_curve_type,
	uint32_t *i_curve_vertex_pos,

	int n_curves, int n_fragments,

	float2* vs, float* arcw1s, uint32_t *curve_path_id,

	int w, int h,
	int* cidx,
	uint64_t* is_path_visible
	) {

	int thread_index = GET_ID();
	if (thread_index >= n_curves) { return; }

	// ??? not necessary ???
	//int curve_index = i_curve_index[thread_index];
	int curve_index = thread_index;

	int i = thread_index, i0 = i;
	int p0 = 0;
	int mode = 0;

	//we're able to deal with inflections - there are at most 4 cut points
	//do it in shared memory first - simpler
	//this uses up almost all the shared memory, hopefully it's worth it
	//POINT_COORDS(8) and POINT_COORDS(9) are temporary slots for the merge sort
	__shared__ float shared_t1_queue[5 * BLOCK_SIZE];
	__shared__ float shared_point_coords[10 * BLOCK_SIZE];

	//do it the spliced way to avoid bank conflicts
	float* t1_queue = shared_t1_queue + threadIdx.x;
	float* point_coords = shared_point_coords + threadIdx.x;
	float arcw1 = 0.f;

	p0 = i_curve_vertex_pos[curve_index];
	mode = i_curve_type[curve_index];
	if (mode == CT_Rational) { arcw1 = arcw1s[curve_index]; }

	//load the vertices
	//we need it when addressing anyway
	//float2* vsp0=vs+p0;
#pragma unroll
	for (int j = 0; j<4; j++) {
		if (j<(mode & 7)) {
			float2 vi = vs[p0 + j];
			POINT_COORDS(j) = d_misalign(vi.x);
			POINT_COORDS(j + 4) = d_misalign(vi.y);
		}
	}

	//compute the monotonic segments
	int n_cuts = 0;
	//bool is_visible = ((int)is_path_visible[curve_path_id[i0]] == 0x01010101);
	bool is_visible = PATH_VISIBLE(is_path_visible[curve_path_id[i0]]);

	float4 q = t_cuts[i0];
	n_cuts = n_cuts_cache[i0];
	T1_QUEUE(0) = q.x;
	T1_QUEUE(1) = q.y;
	T1_QUEUE(2) = q.z;
	T1_QUEUE(3) = q.w;

	if (is_visible) { T1_QUEUE(n_cuts) = 1.f; n_cuts++; }

	////////////////
	//loop over the segments and do the deed
	//int cidx_i0 = cidx[i0];
	int cidx_i0 = curve_index;
	TEMPLATIFY_MODES(
		(k_make_intersection_impl_1<mode, FRAG_SIZE, BLOCK_SIZE>(
			cidx_i0, arcw1, t1_queue, n_cuts, point_coords, ts, pcnt, n_fragments, w, h));
	);
}

#undef T1_QUEUE
#undef POINT_COORDS

// -------- -------- -------- -------- -------- -------- -------- --------

//texture<int,2> tex_amask;
//texture<int,2> tex_pmask;
//texture<int,1> tex_amask;
//texture<int,1> tex_pmask;
template <int MSIZE>
__forceinline__ __device__ int2 getHammersleyMaskInVectorTypes(
	const float2 &p_base,
	const float2 &p0, const float2 &p1,
	int *amaskTable, int *pmaskTable) {

#define NORMALIZE_SEG_DIRECTION
#ifdef NORMALIZE_SEG_DIRECTION
	int x0, y0, x1, y1;

	if (p0.x > p1.x) {
		x0 = __float2int_rn((p0.x - p_base.x) * MSIZE);
		y0 = __float2int_rn((p0.y - p_base.y) * MSIZE);

		x1 = __float2int_rn((p1.x - p_base.x) * MSIZE);
		y1 = __float2int_rn((p1.y - p_base.y) * MSIZE);
	}
	else {
		x1 = __float2int_rn((p0.x - p_base.x) * MSIZE);
		y1 = __float2int_rn((p0.y - p_base.y) * MSIZE);

		x0 = __float2int_rn((p1.x - p_base.x) * MSIZE);
		y0 = __float2int_rn((p1.y - p_base.y) * MSIZE);
	}
#else
	int x0 = __float2int_rn((p0.x - p_base.x) * MSIZE);
	int y0 = __float2int_rn((p0.y - p_base.y) * MSIZE);

	int x1 = __float2int_rn((p1.x - p_base.x) * MSIZE);
	int y1 = __float2int_rn((p1.y - p_base.y) * MSIZE);
#endif

	x0 = limit<0, MSIZE>(x0);
	x1 = limit<0, MSIZE>(x1);
	y0 = limit<0, MSIZE>(y0);
	y1 = limit<0, MSIZE>(y1);

	const int ss = MSIZE + 1;

	int amask = amaskTable[y0 * ss + y1];
	int pmask = pmaskTable[(y0 * ss + x0) * (ss * ss) + (y1 * ss) + x1];
	//int amask = tex1Dfetch(tex_amask,y0 * ss + y1);
	//int pmask = tex1Dfetch(tex_pmask,(y0 * ss + x0) * (ss * ss) + (y1 * ss) + x1);

	return make_int2(pmask, amask);
}

// -------- -------- -------- -------- -------- -------- -------- --------

//template<int A, int M, int MSIZE>
template<int FRAG_SIZE, int BLOCK_SIZE>
__global__ void k_gen_fragment(
	int n_paths,
	int n_curves,
	int n_fragments,
	int stride_fragments,

	int2 * ts,

	uint32_t *i_curve_path_id,
	uint32_t *i_curve_vertex_pos,
	uint8_t *i_curve_type,
	uint8_t *i_curve_reversed,

	float *i_curve_arc_w,
	float2* vs,

	int width, int height, 
	uint32_t *amaskTable,
	uint32_t *pmaskTable,

	int * o_frag_data
	) {

	int thread_index = GET_ID();
	//int frag_index = thread_index;

	int i = thread_index;
	if (i >= n_fragments) { return; }

	int2 cid_t0 = ts[i];
	int2 cid_t1 = (i + 1) != n_fragments ? ts[i + 1] : make_int2(-1, 0x3f800000);

	int int_t0 = cid_t0.y;
	int int_t1 = cid_t1.y;

	auto end_flag_0 = int_t0 & 2;
	auto end_flag_1 = int_t1 & 2;

	auto intersection_flag_0 = int_t0 & 1;
	auto intersection_flag_1 = int_t1 & 1;

	float t0 = __int_as_float(cid_t0.y & 0xFFFFFFFC);
	float t1 = __int_as_float(cid_t1.y & 0xFFFFFFFC);
	int cid0 = cid_t0.x;
	int cid1 = cid_t1.x;
	int cid = cid0;
	int pathid = i_curve_path_id[cid0];
	int yx = 0xFFFEFFFE;
	int winding_number = 0;
	if (cid0 != cid1) { t1 = 1.f; int_t1 = __float_as_int(t1); }
	//int curve_shift_flag = 0;

	if (t0 < t1) {

		float2 ov0, ov1, ov2, ov3;
		//float ovw;

		float2 v0 = make_float2(-1.f, -1.f);
		float2 v1 = make_float2(-1.f, -1.f);
		float2 v2 = make_float2(-1.f, -1.f);
		float2 v3 = make_float2(-1.f, -1.f);
		//float vw = 1.f;

		float2 cut1 = make_float2(-1.f, -1.f);
		float2 tmp;

		//float f0a = 0.f, f1a = 0.f;
		int2 f0f1m = make_int2(0, 0);

		uint8_t rev = i_curve_reversed[cid];
		auto curve_type = i_curve_type[cid];
		auto curve_v_pos = i_curve_vertex_pos[cid];

		auto p0 = curve_v_pos;

		//integration code taken from [Manson and Schaefer 2013]
		switch (curve_type) {
		default: {
			ASSERT(0);
			return;
		}
		case CT_Linear:
		{
			ov0 = vs[p0 + 0]; ov1 = vs[p0 + 1];
			v0 = make_float2(::lerp(ov0.x, ov1.x, t0), ::lerp(ov0.y, ov1.y, t0));
			v1 = make_float2(::lerp(ov0.x, ov1.x, t1), ::lerp(ov0.y, ov1.y, t1));
			if (rev) { float2 tmp = v0; v0 = v1; v1 = tmp; }
			cut1 = v1;
			//float x_base = floorf((v0.x + cut1.x)*0.5f);
			//if (A) {
			//	f0a = 0.5f*((v0.x - x_base) + (v1.x - x_base))*(v1.y - v0.y);
			//}
			break;
		}
		case CT_Quadratic:
		{
			ov0 = vs[p0 + 0]; ov1 = vs[p0 + 1]; ov2 = vs[p0 + 2];

			float2 la0 = lerp2(ov0, ov1, t0);
			float2 la1 = lerp2(ov1, ov2, t0);
			float2 lb0 = lerp2(ov0, ov1, t1);
			float2 lb1 = lerp2(ov1, ov2, t1);
			v0 = lerp2(la0, la1, t0);
			v1 = lerp2(la0, la1, t1);
			v2 = lerp2(lb0, lb1, t1);
			if (rev) { float2 tmp = v0; v0 = v2; v2 = tmp; }
			cut1 = v2;
			//if (A) {
			//	float x_base = floorf((v0.x + cut1.x)*0.5f);
			//	float y_base = floorf((v0.y + cut1.y)*0.5f);
			//	float p00 = v0.x - x_base;
			//	float p01 = v0.y - y_base;
			//	float p10 = v1.x - x_base;
			//	float p11 = v1.y - y_base;
			//	float p20 = v2.x - x_base;
			//	float p21 = v2.y - y_base;
			//	f0a = (1.0f / 6.0f)*((-2.0f*p11*p20) + (-1.0f*p01*((2.0f*p10) + p20))
			//		+ (2.0f*p10*p21) + (3.0f*p20*p21) + (p00*((-3.0f*p01) + (2.0f*p11) + p21)));
			//}
			break;
		}
		case CT_Cubic:
		{
			ov0 = vs[p0 + 0]; ov1 = vs[p0 + 1]; ov2 = vs[p0 + 2]; ov3 = vs[p0 + 3];

			////////
			//const bool A = true;
			const bool A = false;
			if (A) {
				float a = t1;
				float b = t0 / t1;
				if (b != b) { b = 0.f; }

				// left
				float2 c30 = lerp2(ov0, ov1, a);
				float2 c31 = lerp2(ov1, ov2, a);
				float2 c32 = lerp2(ov2, ov3, a);

				float2 c20 = lerp2(c30, c31, a);
				float2 c21 = lerp2(c31, c32, a);

				v0 = ov0;
				v1 = c30;
				v2 = c20;
				v3 = lerp2(c20, c21, a);

				// right
				ov0 = v0;
				ov1 = v1;
				ov2 = v2;
				ov3 = v3;

				c30 = lerp2(ov0, ov1, b);
				c31 = lerp2(ov1, ov2, b);
				c32 = lerp2(ov2, ov3, b);

				c20 = lerp2(c30, c31, b);
				c21 = lerp2(c31, c32, b);

				v0 = lerp2(c20, c21, b);
				v1 = c21;
				v2 = c32;
				v3 = ov3;
			}
			else {
				float2 c30 = lerp2(ov0, ov1, t0);
				float2 c31 = lerp2(ov1, ov2, t0);
				float2 c32 = lerp2(ov2, ov3, t0);
				float2 c20 = lerp2(c30, c31, t0);
				float2 c21 = lerp2(c31, c32, t0);
				v0 = lerp2(c20, c21, t0);
				c30 = lerp2(ov0, ov1, t1);
				c31 = lerp2(ov1, ov2, t1);
				c32 = lerp2(ov2, ov3, t1);
				c20 = lerp2(c30, c31, t1);
				c21 = lerp2(c31, c32, t1);
				v3 = lerp2(c20, c21, t1);
			}

			if (rev) {
				tmp = v0; v0 = v3; v3 = tmp;
				tmp = v1; v1 = v2; v2 = tmp;
			}
			cut1 = v3;
			//if (A) {
			//	float x_base = floorf((v0.x + cut1.x)*0.5f);
			//	float y_base = floorf((v0.y + cut1.y)*0.5f);
			//	float p00 = v0.x - x_base;
			//	float p01 = v0.y - y_base;
			//	float p10 = v1.x - x_base;
			//	float p11 = v1.y - y_base;
			//	float p20 = v2.x - x_base;
			//	float p21 = v2.y - y_base;
			//	float p30 = v3.x - x_base;
			//	float p31 = v3.y - y_base;
			//	f0a = ((1.0f / 20.0f)*((-3.0f*p11*p20) + (-3.0f*p11*p30)
			//		+ (-6.0f*p21*p30) + (-1.0f*p01*((6.0f*p10) + (3.0f*p20) + p30))
			//		+ (6.0f*p20*p31) + (10.0f*p30*p31) + (3.0f*p10*(p21 + p31))
			//		+ (p00*((-10.0f*p01) + (6.0f*p11) + (3.0f*p21) + p31))));
			//}
			break;
		}
		case CT_Rational:
		{
			//cut and pbuf / abuf - rational
			float arcw1 = i_curve_arc_w[cid0];
			ov0 = vs[p0 + 0]; ov1 = vs[p0 + 1]; ov2 = vs[p0 + 2];
			ov1 *= arcw1;

			float2 la0 = lerp2(ov0, ov1, t0);
			float2 la1 = lerp2(ov1, ov2, t0);
			float2 lb0 = lerp2(ov0, ov1, t1);
			float2 lb1 = lerp2(ov1, ov2, t1);
			v0 = lerp2(la0, la1, t0);
			v1 = lerp2(la0, la1, t1);
			v2 = lerp2(lb0, lb1, t1);
			float w0 = (1 - t0)*(1 - t0) + 2.f*(1 - t0)*t0*arcw1 + t0*t0;
			//float w1 = (1 - t0)*(1 - t1) + ((1 - t0)*t1 + (1 - t1)*t0)*arcw1 + t0*t1;
			float w2 = (1 - t1)*(1 - t1) + 2.f*(1 - t1)*t1*arcw1 + t1*t1;

			v0 *= safeRcp(w0);
			v2 *= safeRcp(w2);
			if (rev) {
				tmp = v0; v0 = v2; v2 = tmp;
				auto tf = w0; w0 = w2; w2 = tf;
			}
			cut1 = v2;
			//if (A) {
			//	float x_base = floorf((v0.x + cut1.x)*0.5f);
			//	float y_base = floorf((v0.y + cut1.y)*0.5f);
			//	v1 *= safeRcp(w1);
			//	float p00 = v0.x - x_base;
			//	float p01 = v0.y - y_base;
			//	float p10 = v1.x - x_base;
			//	float p11 = v1.y - y_base;
			//	float p20 = v2.x - x_base;
			//	float p21 = v2.y - y_base;
			//	float aaa = w0*w2 - w1*w1;
			//	if (aaa > 0.001f) {
			//		float sqrt_aaa = sqrtf(aaa);
			//		f0a = (sqrt_aaa*((p01*p10 + p00*(p01 - p11) + p11*p20 - (p10 + p20)*p21)*w1*w1 - (p00 + p20)*(p01 - p21)*w0*w2) +
			//			(p01*(p10 - p20) + p11*p20 - p10*p21 + p00*(-p11 + p21))*w0*w1*w2*(atan2f((-w0 + w1), sqrt_aaa) + atan2f((w1 - w2), sqrt_aaa))) /
			//			(2.f*sqrt_aaa*aaa);
			//	}
			//	else {
			//		// If the weights are too close to constant the rational formula breaks, so revert to quadratic bezier.
			//		f0a = ((1.0f / 6.0f)*((-2.0f*p11*p20) + (-1.0f*p01*((2.0f*p10) + p20)) + (2.0f*p10*p21) + (3.0f*p20*p21) + (p00*((-3.0f*p01) + (2.0f*p11) + p21))));
			//	}
			//}
			break;
		}
		} // end of switch.

		if (rev) {
			int t;
			t = end_flag_0; end_flag_0 = end_flag_1; end_flag_1 = t;
			t = intersection_flag_0; intersection_flag_0 = intersection_flag_1; intersection_flag_1 = t;
		}

		v1 = cut1;

		float2 av0 = v0;
		float2 av1 = v1;

		//if (!end_flag_0) {
		//	if (intersection_flag_0 == 0) { av0.x = round(av0.x); }
		//	else { av0.y = round(av0.y); }
		//}
		//if (!end_flag_1) {
		//	if (intersection_flag_1 == 0) { av1.x = round(av1.x); }
		//	else { av1.y = round(av1.y); }
		//}

		int raw_frag_x = d_float2int_rd<FRAG_SIZE>((av0.x + av1.x)*0.5f);
		int raw_frag_y = d_float2int_rd<FRAG_SIZE>((av0.y + av1.y)*0.5f);

		int x = min(max(raw_frag_x, -FRAG_SIZE), (int)((width & 0xFFFFFFFE) + 2));
		int y = min(max(raw_frag_y, -FRAG_SIZE), (int)((height & 0xFFFFFFFE) + 2));

		//if (y == 574) {
		//	printf("%d %d %e %e (%f %f) (%f %f)\n", thread_index, curve_type, t0, t1, v0.x, v1.x, v0.y, v1.y);
		//}

		//if (y == 618) {
		//	printf("(%d,%d) (%f,%f) (%f,%f)\n", x, y, v0.x, v0.y, v1.x, v1.y);
		//}

		int y_shift = (y + 0x7FFF) << 16;
		int x_shift = (x + 0x7FFF) & 0xFFFF;

		if (0 <= raw_frag_y && raw_frag_y < height) { yx = y_shift | x_shift; }

		// winding number
		int wn_y = y + 1;

		winding_number = 0;
		if (v0.y == v1.y) {
			winding_number = 0;
		}
		else {
			if (v0.y <= wn_y && wn_y < v1.y) {
				winding_number = -1;
			}
			else if (v1.y <= wn_y && wn_y < v0.y) {
				winding_number = 1;
			}
		}

	}

	/*
	output
	----------------------------------------------------------------
	| offset  | size  |
	| sf * 0  | nf    | position
	| sf * 1  | nf    | index
	| sf * 2  | nf    | pathid
	| sf * 3  | npath | sort segment
	| sf * 4  | nf    | winding number
	| sf * 5  | -     | -
	| sf * 6  | -     | -
	| sf * 7  | -     | -
	| sf * 8  | nf    | frag curve_id
	| sf * 9  | nf    | -
	| sf * 10 | nf    | scan segment
	| sf * 11 | -     | -
	*/

	auto o_pos = o_frag_data;
	auto o_sort_index = o_frag_data + stride_fragments;
	auto o_path_id = o_frag_data + stride_fragments * 2;
	auto o_winding_number = o_frag_data + stride_fragments * 4;
	auto o_frag_curve_id = o_frag_data + stride_fragments * 8;

	auto o_sort_segment = o_frag_data + stride_fragments * 3;
	auto o_scan_segment = o_frag_data + stride_fragments * 10;

	float *o_frag_t0 = (float*)o_frag_data + stride_fragments * 11;
	float *o_frag_t1 = (float*)o_frag_data + stride_fragments * 12;

	o_pos[i] = yx;
	o_sort_index[i] = i;
	o_path_id[i] = pathid;
	o_winding_number[i] = winding_number;

	//o_frag_curve_id[i] = (side_1 << 31) | (side_0 << 30) | cid0;
	//o_frag_curve_id[i] = (curve_shift_flag << 31) | cid0;
	o_frag_curve_id[i] = cid0;

	o_frag_t0[i] = __int_as_float(int_t0);
	o_frag_t1[i] = __int_as_float(int_t1);

	// mask or analytical value.
	//((int2*)pfragi)[i + (stride_fragments >> 1) * 5] = make_int2(f0i, f1i);

	// sort segment.
	int scanseg = 0;

	if (cid0 != cid1) {
		int pathid1 = 0;
		if (cid1 >= 0) { pathid1 = i_curve_path_id[cid1]; }
		else { pathid1 = n_paths; }
		scanseg = (pathid < pathid1);
		for (int j = pathid + 1; j <= pathid1; j++) {
			o_sort_segment[j] = i + 1;
		}
	}
	else {
		scanseg = 0;
	}
	o_scan_segment[i + 1] = scanseg;
	if (!i) {
		for (int j = 0; j <= pathid; j++) { o_sort_segment[j] = 0; }
		o_scan_segment[0] = 1;
		o_pos[n_fragments] = -1;
	}

}

// -------- -------- -------- -------- -------- -------- -------- --------

template <int MSIZE>
__device__ inline uint2 d_get_hammersley_mask(
	const float2 &p_base,
	const float2 &p0, const float2 &p1,
	uint *amaskTable, uint *pmaskTable) {

#define NORMALIZE_SEG_DIRECTION
#ifdef NORMALIZE_SEG_DIRECTION
	int x0, y0, x1, y1;

	if (p0.x > p1.x) {
		x0 = __float2int_rn((p0.x - p_base.x) * MSIZE);
		y0 = __float2int_rn((p0.y - p_base.y) * MSIZE);

		x1 = __float2int_rn((p1.x - p_base.x) * MSIZE);
		y1 = __float2int_rn((p1.y - p_base.y) * MSIZE);
	}
	else {
		x1 = __float2int_rn((p0.x - p_base.x) * MSIZE);
		y1 = __float2int_rn((p0.y - p_base.y) * MSIZE);

		x0 = __float2int_rn((p1.x - p_base.x) * MSIZE);
		y0 = __float2int_rn((p1.y - p_base.y) * MSIZE);
	}
#else
	int x0 = __float2int_rn((p0.x - p_base.x) * MSIZE);
	int y0 = __float2int_rn((p0.y - p_base.y) * MSIZE);

	int x1 = __float2int_rn((p1.x - p_base.x) * MSIZE);
	int y1 = __float2int_rn((p1.y - p_base.y) * MSIZE);
#endif

	x0 = limit<0, MSIZE>(x0);
	x1 = limit<0, MSIZE>(x1);
	y0 = limit<0, MSIZE>(y0);
	y1 = limit<0, MSIZE>(y1);

	const int ss = MSIZE + 1;

	//uint32_t amask = amaskTable[y0 * ss + y1];
	//uint32_t pmask = pmaskTable[(y0 * ss + x0) * (ss * ss) + (y1 * ss) + x1];

	uint32_t amask = __ldg(amaskTable + (y0 * ss + y1));
	uint32_t pmask = __ldg(pmaskTable + ((y0 * ss + x0) * (ss * ss) + (y1 * ss) + x1));

	//int amask = tex1Dfetch(tex_amask,y0 * ss + y1);
	//int pmask = tex1Dfetch(tex_pmask,(y0 * ss + x0) * (ss * ss) + (y1 * ss) + x1);

	return make_uint2(pmask, amask);
}

// -------- -------- -------- -------- -------- -------- -------- --------
template <int MSIZE>
__device__ inline uint32_t d_get_hammersley_a_mask( 
	const float2 &p_base, const float2 &p0, const float2 &p1, uint32_t *amaskTable) {

	int y0, y1;

	if (p0.x > p1.x) {
		y0 = __float2int_rn((p0.y - p_base.y) * MSIZE);
		y1 = __float2int_rn((p1.y - p_base.y) * MSIZE);
	}
	else {
		y1 = __float2int_rn((p0.y - p_base.y) * MSIZE);
		y0 = __float2int_rn((p1.y - p_base.y) * MSIZE);
	}

	y0 = limit<0, MSIZE>(y0);
	y1 = limit<0, MSIZE>(y1);

	const int ss = MSIZE + 1;

	//uint32_t amask = amaskTable[y0 * ss + y1];
	uint32_t amask = __ldg(amaskTable + (y0 * ss + y1));
	//int amask = tex1Dfetch(tex_amask,y0 * ss + y1);

	return amask;
}

// -------- -------- -------- -------- -------- -------- -------- --------

} // end of namespace Rasterizers

} // end of namespace Mochimazui

#endif