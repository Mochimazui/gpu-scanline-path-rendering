
#ifndef _VG_RASTERIZER_QM_MASK_H_
#define _VG_RASTERIZER_QM_MASK_H_

#include <cuda.h>

#include <mochimazui/cuda_array.h>

#include "ras_define.h"
#include "ras_device_func.h"

#include "../../cutil_math.h"

namespace Mochimazui {

// -------- -------- -------- -------- -------- -------- -------- --------
namespace QM_Mask_Sample_Position {

extern float mpvg_8_x[8];
extern float mpvg_8_y[8];

extern float mpvg_32_x[32];
extern float mpvg_32_y[32];

std::vector<float2> mpvg_sample_position(int n);
std::vector<float2> gl_sample_position(int n);
std::vector<float2> vg_sample_position(int n);

}

// -------- -------- -------- -------- -------- -------- -------- --------
namespace QM_Mask_1x1 {

//#define VERTICAL_TABLE_RES 129
#define TABLE_RES 256
#define PACKING_SCALE 0.5f
//#define N_SAMPLES 128

//really give 128 samples here
static __global__ void makeTable(int* ret8, int* ret32, float2* samples) {
	int i = GET_ID();
	int y = (int)((unsigned int)i / (unsigned int)TABLE_RES);
	int x = i - y*TABLE_RES;
	//xf,yf: raw float2 texture coordinate
	float xf = ((float)x + 0.5f)*(1.f / (float)TABLE_RES);
	float yf = ((float)y + 0.5f)*(1.f / (float)TABLE_RES);
	//convert to line equation: dot(N,P)=C, P is the pixel coordinate
	float2 N_rev = 2.f*make_float2(xf - 0.5f, yf - 0.5f);
	float lg2_rev = dot(N_rev, N_rev);
	float lg_rev = sqrtf(lg2_rev);
	float2 N = (1.f / lg_rev)*N_rev;
	//the worst precision is at the boundary, about (1-sqrt(2)*0.5)=0.3
	//origin is defined at (0.5,0.5) here
	float C = max(1.f - lg_rev, 0.f)*(1.f / PACKING_SCALE);
	if (N.x < 0.f) {
		//make N point to the right
		N = -N;
		C = -C;
	}
	//hack for the worst
	if (x == 37 && y == 218) {
		N = normalize(make_float2(1.f, -1.f));
		C = 0.f;
	}
	//translate origin to (0,0)
	C += 0.5f*(N.x + N.y);

	//test the samples
	int mask_8 = 0;
	for (int j = 0; j<8; j++) {
		float2 P = samples[j];
		if (dot(N, P) - C>0.f) {
			mask_8 |= (1 << j);
		}
	}

	int mask_acc = 0;
	{
		int d = 0;
		for (int k = 0; k<32; k++) {
			float2 P = samples[8 + k];
			if (dot(N, P) - C>0.f) {
				d |= (1 << k);
			}
		}
		mask_acc = d;
	}

	//printf("%08x %08x %08x %08x\n",mask_acc[0], mask_acc[1], mask_acc[2], mask_acc[3]);
	//printf("%08x\n",mask_8);
	ret8[i] = mask_8;
	ret32[i] = mask_acc;
}

static texture<int32_t, 2, cudaReadModeElementType> tex_pixel_table8;
static texture<int32_t, 2, cudaReadModeElementType> tex_pixel_table32;

#define FETCH_TEST_RES 33

template<int N_SAMPLES>
static __forceinline__ __device__ int32_t fetchPixelTable(float2 P0, float2 P1) {
	if (P1.y < P0.y) { Rasterizer::swap(P0, P1); }
	float2 D = P1 - P0;
	float2 N = normalize(make_float2(D.y, -D.x));
	float C = dot(N, P0);
	//translate origin to (0.5,0.5)
	C -= 0.5f*(N.x + N.y);
	float C_sign = __int_as_float((__float_as_int(C) & 0x80000000) + 0x3f800000);
	/*if (C<0.f) {
	N = -N;
	C = -C;
	}*/
	//reverse the distance
	float2 N_rev = max(1.f - C*C_sign*PACKING_SCALE, 0.f)*C_sign*N;
	float2 uv = N_rev*0.5f + make_float2(0.5f, 0.5f);
	if (N_SAMPLES == 8) {
		return tex2D(tex_pixel_table8, uv.x, uv.y);
	}
	else {
		return tex2D(tex_pixel_table32, uv.x, uv.y);
	}
}

template<int N_SAMPLES>
static __forceinline__ __host__ __device__ int fetchVerticalTable(float y) {
	if (N_SAMPLES == 8) {
		uint32_t mask = 0xFF;
		mask >>= (8 - (int)round(y * 8));
		return mask;
	}
	else {
		uint32_t mask = 0xFFFFFFFF;
		mask >>= (32 - (int)round(y * 32));
		return mask;
	}
}

template<int N_SAMPLES>
static __forceinline__ __host__ __device__ int fetchVerticalTable2(float y0, float y1) {
	if (N_SAMPLES == 8) {
		int shift_0 = (8 - (int)(y0 * 8 + 0.5));
		int shift_1 = (8 - (int)(y1 * 8 + 0.5));
		shift_0 = max(shift_0, 0);
		shift_1 = max(shift_1, 0);
		uint32_t mask = 0xFF;
		mask = (mask >> shift_0) ^ (mask >> shift_1);
		return mask;
	}
	else {
		int shift_0 = (32 - (int)(y0 * 32 + 0.5));
		int shift_1 = (32 - (int)(y1 * 32 + 0.5));
		shift_0 = max(shift_0, 0);
		shift_1 = max(shift_1, 0);
		uint32_t mask = 0xFFFFFFFF;
		mask = (mask >> shift_0) ^ (mask >> shift_1);
		return mask;
	}
}

static __global__ void testFetchTable_32(float2* samples) {
	int i = GET_ID();
	int p1 = i / (FETCH_TEST_RES*FETCH_TEST_RES);
	int p0 = i - p1*(FETCH_TEST_RES*FETCH_TEST_RES);
	int y0 = p0 / FETCH_TEST_RES, x0 = p0 - y0*FETCH_TEST_RES;
	int y1 = p1 / FETCH_TEST_RES, x1 = p1 - y1*FETCH_TEST_RES;
	if (y0 == y1) { return; }
	float2 P0 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x0, (float)y0);
	float2 P1 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x1, (float)y1);

	int mask_out = fetchPixelTable<32>(P0, P1);
	int mask_acc[1];
#pragma unroll
	for (int j = 0; j < 1; j++) {
		int d = 0;
		for (int k = 0; k < 32; k++) {
			float2 P = samples[j * 32 + k];
			if (P.x > (P.y - P0.y) / (P1.y - P0.y)*(P1.x - P0.x) + P0.x) {
				d |= (1 << k);
			}
		}
		mask_acc[j] = d;
	}
	int err = __popc(mask_out^mask_acc[0]);
	if (err > 8) {
		float2 D = P1 - P0;
		float2 N = normalize(make_float2(D.y, -D.x));
		float C = dot(N, P0);
		//translate origin to (0.5,0.5)
		C -= 0.5f*(N.x + N.y);
		if (C < 0.f) {
			N = -N;
			C = -C;
		}
		//reverse the distance
		float C2 = max(1.f - C*PACKING_SCALE, 0.f);
		float2 N_rev = C2*N;
		float2 uv = N_rev*0.5f + make_float2(0.5f, 0.5f);
		printf("32 - big error %.2f,%.2f - %.2f,%.2f  C2=%.3f %d,%d %d\n", P0.x, P0.y, P1.x, P1.y, C2,
			int(float(TABLE_RES)*uv.x), int(float(TABLE_RES)*uv.y), err);
	}
	else {
		//printf("OK\n");
	}
}

static __global__ void testFetchTable_8(float2* samples) {
	int i = GET_ID();
	int p1 = i / (FETCH_TEST_RES*FETCH_TEST_RES);
	int p0 = i - p1*(FETCH_TEST_RES*FETCH_TEST_RES);
	int y0 = p0 / FETCH_TEST_RES, x0 = p0 - y0*FETCH_TEST_RES;
	int y1 = p1 / FETCH_TEST_RES, x1 = p1 - y1*FETCH_TEST_RES;
	if (y0 == y1) { return; }
	float2 P0 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x0, (float)y0);
	float2 P1 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x1, (float)y1);

	int mask_out = fetchPixelTable<8>(P0, P1);
	int mask_acc = 0;

#pragma unroll
	for (int j = 0; j < 1; j++) {
		int d = 0;
		for (int k = 0; k < 8; k++) {
			float2 P = samples[j * 8 + k];
			if (P.x > (P.y - P0.y) / (P1.y - P0.y)*(P1.x - P0.x) + P0.x) {
				d |= (1 << k);
			}
		}
		mask_acc = (mask_acc << 8) | d;
	}

	int err = __popc(mask_out^mask_acc);
	if (err >= 4) {
		float2 D = P1 - P0;
		float2 N = normalize(make_float2(D.y, -D.x));
		float C = dot(N, P0);
		//translate origin to (0.5,0.5)
		C -= 0.5f*(N.x + N.y);
		if (C < 0.f) {
			N = -N;
			C = -C;
		}
		//reverse the distance
		float C2 = max(1.f - C*PACKING_SCALE, 0.f);
		float2 N_rev = C2*N;
		float2 uv = N_rev*0.5f + make_float2(0.5f, 0.5f);
		printf("8 - big error %.2f,%.2f - %.2f,%.2f  C2=%.3f %d,%d %d\n", P0.x, P0.y, P1.x, P1.y, C2,
			int(float(TABLE_RES)*uv.x), int(float(TABLE_RES)*uv.y), err);
	}
	else {
		//printf("OK\n");
	}
}

static void init_qm_mask_table_1x1(
	int n_samples,
	float2 * gl_mask_8,
	float2 * gl_mask_32,
	CUDATL::CUDAArray<float2> &_sample_position,
	CUDATL::CUDAArray<int> &_qm_mask_table_pixel8,
	CUDATL::CUDAArray<int> &_qm_mask_table_pixel32
	) {

	float2 gen_fsample_8[8];
	float2 gen_fsample_32[32];

	if (!gl_mask_32) {

		//Mochimazui::Bitmap bmp;
		//bmp.resize(32, 32);
		//bmp.fill(Mochimazui::u8rgba(0, 0, 0, 255));

		int2 isample_32[32];
		double gap = 1.0 / 32;
		double hgap = gap / 2;

		for (int i = 0; i < 32; ++i) {

			int bc = 0;
			int x = i;
			int y = 0;

			while (x) {
				y <<= 1;
				y |= x & 1;
				x >>= 1;
				++bc;
			}

			int scale = 1 << bc;
			double f = y / (float)scale;

			int yy = (int)(f * 32);

			auto &s = isample_32[i];
			s.x = yy;
			s.y = i;

			//bmp.setPixel(s.x, s.y, u8rgba(0, 255, 0, 255));

			auto &sp = gen_fsample_32[i];
			sp.x = (float)(s.x * gap + hgap);
			sp.y = (float)(s.y * gap + hgap);
		}

		//bmp.save("mask_32.bmp");

		gl_mask_32 = gen_fsample_32;
	}

	if (!gl_mask_8) {

		//Mochimazui::Bitmap bmp;
		//bmp.resize(8, 8);
		//bmp.fill(Mochimazui::u8rgba(0, 0, 0, 255));

		int2 isample_8[8];
		double gap = 1.0 / 8;
		double hgap = gap / 2;

		for (int i = 0; i < 8; ++i) {

			int bc = 0;
			int x = i;
			int y = 0;

			while (x) {
				y <<= 1;
				y |= x & 1;
				x >>= 1;
				++bc;
			}

			int scale = 1 << bc;
			double f = y / (float)scale;

			int yy = (int)(f * 8);

			auto &s = isample_8[i];
			s.x = yy;
			s.y = i;

			//bmp.setPixel(s.x, s.y, u8rgba(0, 255, 0, 255));

			auto &sp = gen_fsample_8[i];
			sp.x = (float)(s.x * gap + hgap);
			sp.y = (float)(s.y * gap + hgap);
		}

		//bmp.save("mask_8.bmp");

		gl_mask_8 = gen_fsample_8;
	}

	float2 fsample_8[8];
	float2 fsample_32[32];

	for (int i = 0; i < 8; ++i) {
		auto sp = gl_mask_8[i];
		fsample_8[i] = sp;
	}

	for (int i = 0; i < 32; ++i) {
		auto sp = gl_mask_32[i];
		fsample_32[i] = sp;
	}

	// -------- -------- -------- --------
	if (n_samples == 8) {
		_sample_position.resizeWithoutCopy(8);
		cudaMemcpy(_sample_position.gptr(), fsample_8, 8 * sizeof(float2),
			cudaMemcpyHostToDevice);
	}
	else {
		_sample_position.resizeWithoutCopy(32);
		cudaMemcpy(_sample_position.gptr(), fsample_32, 32 * sizeof(float2),
			cudaMemcpyHostToDevice);
	}

	// -------- -------- -------- --------
	_qm_mask_table_pixel8.resizeWithoutCopy(TABLE_RES*TABLE_RES);
	int *table_pixel8 = _qm_mask_table_pixel8.gptr();
	_qm_mask_table_pixel32.resizeWithoutCopy(TABLE_RES*TABLE_RES);
	int *table_pixel32 = _qm_mask_table_pixel32.gptr();

	float2* samples = NULL;
	cudaMalloc((void**)&samples, (8 + 32)*sizeof(float2));

	cudaMemcpy(samples, fsample_8, 8 * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(samples + 8, fsample_32, 32 * sizeof(float2), cudaMemcpyHostToDevice);
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("before makeTable");
	LAUNCH(makeTable, TABLE_RES*TABLE_RES, 256, (table_pixel8, table_pixel32, samples));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("after makeTable");

	size_t offset = 0;
	tex_pixel_table8.normalized = 1;
	tex_pixel_table8.filterMode = cudaFilterModePoint;
	tex_pixel_table8.addressMode[0] = cudaAddressModeClamp;
	tex_pixel_table8.addressMode[1] = cudaAddressModeClamp;
	cudaBindTexture2D(&offset, tex_pixel_table8, table_pixel8, cudaCreateChannelDesc<int>(), TABLE_RES, TABLE_RES, TABLE_RES*sizeof(int));
	assert(!offset);
	offset = 0;
	tex_pixel_table32.normalized = 1;
	tex_pixel_table32.filterMode = cudaFilterModePoint;
	tex_pixel_table32.addressMode[0] = cudaAddressModeClamp;
	tex_pixel_table32.addressMode[1] = cudaAddressModeClamp;
	cudaBindTexture2D(&offset, tex_pixel_table32, table_pixel32, cudaCreateChannelDesc<int>(), TABLE_RES, TABLE_RES, TABLE_RES*sizeof(int));
	assert(!offset);

	//
	//LAUNCH(testFetchTable_8, FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES, 
	//	256, (samples));
	//cudaDeviceSynchronize();
	//LAUNCH(testFetchTable_32, FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES, 
	//	256, (samples + 8));
	//cudaDeviceSynchronize();

	// -------- -------- -------- --------
	cudaFree(samples);
	cudaDeviceSynchronize();
}

}

namespace QM_Mask_2x2 {

//#define VERTICAL_TABLE_RES 129
#define TABLE_RES 256
#define PACKING_SCALE 0.5f
//#define N_SAMPLES 128

//really give 128 samples here
static __global__ void makeTable(int* ret8, int4* ret32, float2* samples) {
	int i = GET_ID();
	int y = (int)((unsigned int)i / (unsigned int)TABLE_RES);
	int x = i - y*TABLE_RES;
	//xf,yf: raw float2 texture coordinate
	float xf = ((float)x + 0.5f)*(1.f / (float)TABLE_RES);
	float yf = ((float)y + 0.5f)*(1.f / (float)TABLE_RES);
	//convert to line equation: dot(N,P)=C, P is the pixel coordinate
	float2 N_rev = 2.f*make_float2(xf - 0.5f, yf - 0.5f);
	float lg2_rev = dot(N_rev, N_rev);
	float lg_rev = sqrtf(lg2_rev);
	float2 N = (1.f / lg_rev)*N_rev;
	//the worst precision is at the boundary, about (1-sqrt(2)*0.5)=0.3
	//origin is defined at (0.5,0.5) here
	float C = max(1.f - lg_rev, 0.f)*(1.f / PACKING_SCALE);
	if (N.x<0.f) {
		//make N point to the right
		N = -N;
		C = -C;
	}
	//hack for the worst
	if (x == 37 && y == 218) {
		N = normalize(make_float2(1.f, -1.f));
		C = 0.f;
	}
	//translate origin to (0,0)
	C += 0.5f*(N.x + N.y);
	//test the samples
	int mask_8 = 0;
	for (int j = 0; j<32; j++) {
		float2 P = samples[j];
		if (dot(N, P) - C>0.f) {
			mask_8 |= (1 << j);
		}
	}
	int mask_acc[4];
#pragma unroll
	for (int j = 0; j<4; j++) {
		int d = 0;
		for (int k = 0; k<32; k++) {
			float2 P = samples[8 * 4 + j * 32 + k];
			if (dot(N, P) - C>0.f) {
				d |= (1 << k);
			}
		}
		mask_acc[j] = d;
	}

	//printf("%08x %08x %08x %08x\n",mask_acc[0], mask_acc[1], mask_acc[2], mask_acc[3]);
	//printf("%08x\n",mask_8);
	ret8[i] = mask_8;
	ret32[i] = make_int4(mask_acc[0], mask_acc[1], mask_acc[2], mask_acc[3]);
}

static texture<int, 2, cudaReadModeElementType> tex_pixel_table8;
static texture<int4, 2, cudaReadModeElementType> tex_pixel_table32;

#define FETCH_TEST_RES 33

//#define ACCURATE_FETCH_TABLE
#ifdef ACCURATE_FETCH_TABLE
template<int N_SAMPLES>
__forceinline__ __device__ int4 fetchPixelTable(
	float2 *sample_position, float2 p0, float2 p1
	) {
	int mask = 0;

	if (p0.y == p1.y) { return make_int4(0, 0, 0, 0); }
	if (p0.y > p1.y) { swap(p0, p1); }

	for (int i = 0; i < 32; ++i) {

		auto s = sample_position[i];

		auto d = p1.y - p0.y;
		auto t = (s.y - p0.y) / d;
		auto x = p0.x * (1 - t) + p1.x * t;

		//if (0.f <= t && t <= 1.f && x <= s.x) {
		if (p0.y < s.y && s.y < p1.y && x < s.x) {
			mask |= 1 << i;
		}
	}

	return make_int4(mask, 0, 0, 0);
}
#else
template<int N_SAMPLES>
static __forceinline__ __device__ int4 fetchPixelTable(float2 P0, float2 P1) {
	//if (P1.x < P0.x) { swap(P0, P1); }
	if (P1.y < P0.y) { Rasterizer::swap(P0, P1); }
	float2 D = P1 - P0;
	float2 N = normalize(make_float2(D.y, -D.x));
	float C = dot(N, P0);
	//translate origin to (0.5,0.5)
	C -= 0.5f*(N.x + N.y);
	float C_sign = __int_as_float((__float_as_int(C) & 0x80000000) + 0x3f800000);
	/*if (C<0.f) {
	N = -N;
	C = -C;
	}*/
	//reverse the distance
	float2 N_rev = max(1.f - C*C_sign*PACKING_SCALE, 0.f)*C_sign*N;
	float2 uv = N_rev*0.5f + make_float2(0.5f, 0.5f);
	if (N_SAMPLES == 8) {
		return make_int4(tex2D(tex_pixel_table8, uv.x, uv.y), 0, 0, 0);
	}
	else {
		return tex2D(tex_pixel_table32, uv.x, uv.y);
	}
}
#endif

template<int N_SAMPLES>
static __forceinline__ __host__ __device__ int4 fetchVerticalTable(float y) {
	if (N_SAMPLES == 8) {
		int mask = 0xffff;
		mask >>= (16 - (int)round(y * 16));
		mask |= mask << 16;
		return make_int4(mask, 0, 0, 0);
	}
	else {
		uint64_t mask = 0xFFFFFFFFFFFFFFFFLL;
		mask >>= (64 - (int)round(y * 64));
		int m0 = mask;
		int m1 = mask >> 32;
		return make_int4(m0, m1, m0, m1);
	}
}

template<int N_SAMPLES>
static __forceinline__ __host__ __device__ int4 fetchVerticalTable2(float y0, float y1) {
	//int4 m0, m1;
	//m0 = fetchVerticalTable(y0);
	//m1 = fetchVerticalTable(y1);
	//return make_int4(m0.x ^ m1.x, m0.y ^ m1.y, m0.z ^ m1.z, m0.w ^ m1.w);
	if (N_SAMPLES == 8) {
		//int shift_0 = (16 - (int)round(y0 * 16));
		//int shift_1 = (16 - (int)round(y1 * 16));
		int shift_0 = (16 - (int)(y0 * 16 + 0.5));
		int shift_1 = (16 - (int)(y1 * 16 + 0.5));
		shift_0 = max(shift_0, 0);
		shift_1 = max(shift_1, 0);

		int mask = 0xffff;
		mask = (mask >> shift_0) ^ (mask >> shift_1);
		mask |= mask << 16;
		return make_int4(mask, 0, 0, 0);
	}
	else {
		//QM: to test - shift by 64 may not work
		//int shift_0 = (64 - (int)round(y0 * 64));
		//int shift_1 = (64 - (int)round(y1 * 64));
		int shift_0 = (64 - (int)(y0 * 64 + 0.5));
		int shift_1 = (64 - (int)(y1 * 64 + 0.5));
		shift_0 = max(shift_0, 0);
		shift_1 = max(shift_1, 0);

		uint64_t mask = 0xFFFFFFFFFFFFFFFFLL;
		mask = (mask >> shift_0) ^ (mask >> shift_1);

		int m0 = mask;
		int m1 = mask >> 32;

		return make_int4(m0, m1, m0, m1);
	}
}

#ifndef ACCURATE_FETCH_TABLE
static __global__ void testFetchTable_32(float2* samples) {
	int i = GET_ID();
	int p1 = i / (FETCH_TEST_RES*FETCH_TEST_RES);
	int p0 = i - p1*(FETCH_TEST_RES*FETCH_TEST_RES);
	int y0 = p0 / FETCH_TEST_RES, x0 = p0 - y0*FETCH_TEST_RES;
	int y1 = p1 / FETCH_TEST_RES, x1 = p1 - y1*FETCH_TEST_RES;
	if (y0 == y1) { return; }
	float2 P0 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x0, (float)y0);
	float2 P1 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x1, (float)y1);

	int4 mask_out = fetchPixelTable<32>(P0, P1);
	int mask_acc[4];
#pragma unroll
	for (int j = 0; j < 4; j++) {
		int d = 0;
		for (int k = 0; k < 32; k++) {
			float2 P = samples[j * 32 + k];
			if (P.x >(P.y - P0.y) / (P1.y - P0.y)*(P1.x - P0.x) + P0.x) {
				d |= (1 << k);
			}
		}
		mask_acc[j] = d;
	}

	int errors[4] = {
		__popc(mask_out.x^mask_acc[0]),
		__popc(mask_out.y^mask_acc[1]),
		__popc(mask_out.z^mask_acc[2]),
		__popc(mask_out.w^mask_acc[3])
	};

	int max_error = max(errors[0], errors[1]);
	max_error = max(max_error, errors[2]);
	max_error = max(max_error, errors[3]);

	if (max_error >= 4) {
		float2 D = P1 - P0;
		float2 N = normalize(make_float2(D.y, -D.x));
		float C = dot(N, P0);
		//translate origin to (0.5,0.5)
		C -= 0.5f*(N.x + N.y);
		if (C < 0.f) {
			N = -N;
			C = -C;
		}

		//reverse the distance
		float C2 = max(1.f - C*PACKING_SCALE, 0.f);
		float2 N_rev = C2*N;
		float2 uv = N_rev*0.5f + make_float2(0.5f, 0.5f);
		//printf("32 - big error %.2f,%.2f - %.2f,%.2f  C2=%.3f %d,%d %d\n", P0.x, P0.y, P1.x, P1.y, C2,
		//	int(float(TABLE_RES)*uv.x), int(float(TABLE_RES)*uv.y), err);

		printf("Fragment(make_float2(%f,%f), make_float2(%f,%f), make_int4(0x%08X,0x%08X,0x%08X,0x%08X),"
			" make_int4(0x%08X,0x%08X,0x%08X,0x%08X)),\n",
			P0.x, P0.y, P1.x, P1.y,
			mask_out.x, mask_out.y, mask_out.z, mask_out.w,
			mask_out.x^mask_acc[0],mask_out.y^mask_acc[1],mask_out.z^mask_acc[2],mask_out.w^mask_acc[3]
			);

	}
	else {
		//printf("OK\n");
	}
}

static __global__ void testFetchTable_8(float2* samples) {
	int i = GET_ID();
	int p1 = i / (FETCH_TEST_RES*FETCH_TEST_RES);
	int p0 = i - p1*(FETCH_TEST_RES*FETCH_TEST_RES);
	int y0 = p0 / FETCH_TEST_RES, x0 = p0 - y0*FETCH_TEST_RES;
	int y1 = p1 / FETCH_TEST_RES, x1 = p1 - y1*FETCH_TEST_RES;
	if (y0 == y1) { return; }
	float2 P0 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x0, (float)y0);
	float2 P1 = 1.f / (FETCH_TEST_RES - 1.f)*make_float2((float)x1, (float)y1);

	int4 mask_out = fetchPixelTable<8>(P0, P1);
	int mask_acc = 0;

#pragma unroll
	for (int j = 0; j < 4; j++) {
		int d = 0;
		for (int k = 0; k < 8; k++) {
			float2 P = samples[j * 8 + k];
			if (P.x >(P.y - P0.y) / (P1.y - P0.y)*(P1.x - P0.x) + P0.x) {
				d |= (1 << k);
			}
		}
		mask_acc = (mask_acc << 8) | d;
	}

	int err = __popc(mask_out.x^mask_acc);
	if (err >= 4) {
		float2 D = P1 - P0;
		float2 N = normalize(make_float2(D.y, -D.x));
		float C = dot(N, P0);
		//translate origin to (0.5,0.5)
		C -= 0.5f*(N.x + N.y);
		if (C < 0.f) {
			N = -N;
			C = -C;
		}
		//reverse the distance
		float C2 = max(1.f - C*PACKING_SCALE, 0.f);
		float2 N_rev = C2*N;
		float2 uv = N_rev*0.5f + make_float2(0.5f, 0.5f);
		printf("8 - big error %.2f,%.2f - %.2f,%.2f  C2=%.3f %d,%d %d\n", P0.x, P0.y, P1.x, P1.y, C2,
			int(float(TABLE_RES)*uv.x), int(float(TABLE_RES)*uv.y), err);
	}
	else {
		//printf("OK\n");
	}
}
#endif

static void init_qm_mask_table(
	int n_samples,
	float2 * gl_mask_8,
	float2 * gl_mask_32,
	CUDATL::CUDAArray<float2> &_sample_position,
	CUDATL::CUDAArray<int> &_qm_mask_table_pixel8,
	CUDATL::CUDAArray<int4> &_qm_mask_table_pixel32
	) {

	std::vector<float2> fsample_8;
	std::vector<float2> fsample_32;

	fsample_8 = QM_Mask_Sample_Position::vg_sample_position(8);
	fsample_32 = QM_Mask_Sample_Position::vg_sample_position(32);

	gl_mask_8 = fsample_8.data();
	gl_mask_32 = fsample_32.data();

	float2 fsample_8x4[32];
	float2 fsample_32x4[128];

	for (int i = 0; i < 8; ++i) {
		auto sp = gl_mask_8[i];
		fsample_8x4[i + 8 * 0] = sp;
		fsample_8x4[i + 8 * 1] = sp + make_float2(0, 1);
		fsample_8x4[i + 8 * 2] = sp + make_float2(1, 0);
		fsample_8x4[i + 8 * 3] = sp + make_float2(1, 1);
	}

	for (int i = 0; i < 32; ++i) {
		fsample_8x4[i] *= .5f;
	}

	for (int i = 0; i < 32; ++i) {
		auto sp = gl_mask_32[i];
		fsample_32x4[i + 32 * 0] = sp;
		fsample_32x4[i + 32 * 1] = sp + make_float2(0, 1);
		fsample_32x4[i + 32 * 2] = sp + make_float2(1, 0);
		fsample_32x4[i + 32 * 3] = sp + make_float2(1, 1);
	}

	for (int i = 0; i < 128; ++i) {
		fsample_32x4[i] *= .5f;
	}

	// -------- -------- -------- --------
	if (n_samples == 8) {
		_sample_position.resizeWithoutCopy(8 * 4);
		cudaMemcpy(_sample_position.gptr(), fsample_8x4, 8 * 4 * sizeof(float2),
			cudaMemcpyHostToDevice);
	}
	else {
		_sample_position.resizeWithoutCopy(32 * 4);
		cudaMemcpy(_sample_position.gptr(), fsample_32x4, 32 * 4 * sizeof(float2),
			cudaMemcpyHostToDevice);
	}

	// -------- -------- -------- --------
	_qm_mask_table_pixel8.resizeWithoutCopy(TABLE_RES*TABLE_RES);
	int *table_pixel8 = _qm_mask_table_pixel8.gptr();
	_qm_mask_table_pixel32.resizeWithoutCopy(TABLE_RES*TABLE_RES);
	int4 *table_pixel32 = _qm_mask_table_pixel32.gptr();

	float2* samples = NULL;
	cudaMalloc((void**)&samples, (8 + 32) * 4 * sizeof(float2));

	cudaMemcpy(samples, fsample_8x4, 8 * 4 * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(samples + 8 * 4, fsample_32x4, 32 * 4 * sizeof(float2), cudaMemcpyHostToDevice);
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("before makeTable");
	LAUNCH(makeTable, TABLE_RES*TABLE_RES, 256, (table_pixel8, table_pixel32, samples));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("after makeTable");

	size_t offset = 0;
	tex_pixel_table8.normalized = 1;
	tex_pixel_table8.filterMode = cudaFilterModePoint;
	tex_pixel_table8.addressMode[0] = cudaAddressModeClamp;
	tex_pixel_table8.addressMode[1] = cudaAddressModeClamp;
	cudaBindTexture2D(&offset, tex_pixel_table8, table_pixel8, cudaCreateChannelDesc<int>(), TABLE_RES, TABLE_RES, TABLE_RES*sizeof(int));
	assert(!offset);
	offset = 0;
	tex_pixel_table32.normalized = 1;
	tex_pixel_table32.filterMode = cudaFilterModePoint;
	tex_pixel_table32.addressMode[0] = cudaAddressModeClamp;
	tex_pixel_table32.addressMode[1] = cudaAddressModeClamp;
	cudaBindTexture2D(&offset, tex_pixel_table32, table_pixel32, cudaCreateChannelDesc<int4>(), TABLE_RES, TABLE_RES, TABLE_RES*sizeof(int4));
	assert(!offset);

	//
	//LAUNCH(testFetchTable_8, FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES, 
	//	256, (samples));
	//LAUNCH(testFetchTable_32, FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES*FETCH_TEST_RES, 
	//	256, (samples + 8 * 4));

	cudaDeviceSynchronize();
	// -------- -------- -------- --------
	cudaFree(samples);
}

}

}

#endif