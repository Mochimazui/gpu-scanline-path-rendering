
#include "ras_cut_mask_comb_scanline.h"

#include <functional>
#include <algorithm>

#include <mochimazui/3rd/helper_cuda.h>
#include <mochimazui/bitmap.h>
#include <mochimazui/stdio_ext.h>

#include "cuda/cuda_cached_allocator.h"
#include "cuda/cuda_sort.h"

#include "../shared/ras_scan.h"
#include "../shared/ras_device_func.h"
#include "../shared/ras_cut.h"
#include "../shared/ras_define.h"

#include "../shared/ras_qm_mask.h"

//#include "../kernel/stroke.h"
//#include "../shared/ras_icurve.cuh"
#include "../kernel/animation.h"

#define ENABLE_WINDING_NUMBER_SEGMENT_SCAN

namespace Mochimazui {

namespace Rasterizer_R_Cut_A_Mask_Comb_Scanline {

using CUDATL::CUDAArray;
using namespace Rasterizer;
using namespace Rasterizer_R_Cut;

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_transformVertex(
	uint32_t i_number,
	float2 *i_vertex,
	uint32_t *i_vertex_path_id,
	float4 m0, float4 m1, float4 m2, float4 m3,
	float w, float h, float z, float cutoff,
	float2 *o_vertex,
	char* is_path_visible) {

	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= i_number) { return; }

	float2 iv2 = ((float2*)i_vertex)[ti];

	float4 iv;
	iv.x = iv2.x; iv.y = iv2.y; iv.z = 0.f; iv.w = 1.f;

	float4 ov;
	ov.x = f4dot(m0, iv);
	ov.y = f4dot(m1, iv);
	ov.z = f4dot(m2, iv);
	ov.w = f4dot(m3, iv);

	ov.x /= ov.w;
	ov.y /= ov.w;
	ov.z /= ov.w;

	if (ov.z < cutoff) { ov.z = cutoff; }
	float w2 = w * .5f;
	float h2 = h * .5f;
	ov.x = (ov.x - w2) * z / (ov.z + z) + w2;
	ov.y = (ov.y - h2) * z / (ov.z + z) + h2;

	// see ras_define.h
	// path visible flag:
	//
	//         y > height
	//
	//         5 | 6 | 7
	// x < 0   3 | x | 4  x > width
	//         0 | 1 | 2 
	//
	//           y < 0
	//

	int x_flag = ov.x < 0 ? 0 : (ov.x <= w ? 1 : 2);
	int y_flag = ov.y < 0 ? 0 : (ov.y <= h ? 1 : 2);

	int pathid = i_vertex_path_id[ti];
	auto path_flag = is_path_visible + pathid * 8;

	switch ((y_flag << 4) | x_flag) {
	case 0x00: path_flag[0] = 1; break;
	case 0x01: path_flag[1] = 1; break;
	case 0x02: path_flag[2] = 1; break;

	case 0x10: path_flag[3] = 1; break;
	case 0x11: path_flag[0] = 1; path_flag[7] = 1; break;
	case 0x12: path_flag[4] = 1; break;

	case 0x20: path_flag[5] = 1; break;
	case 0x21: path_flag[6] = 1; break;
	case 0x22: path_flag[7] = 1; break;

	default: break;
	}

	if (floor(ov.x) == ov.x) { ov.x = ov.x - 1.f / 256.f; }
	((float2*)o_vertex)[ti] = make_float2(ov.x, ov.y);
}

////////////
#define T1_QUEUE(i) (t1_queue[(i)*BLOCK_SIZE])
#define POINT_COORDS(i) (point_coords[(i)*BLOCK_SIZE])

// -------- -------- -------- -------- -------- -------- -------- --------
template<int N_SAMPLES,int FRAG_SIZE, int BLOCK_SIZE>
__global__ void k_gen_fragment_and_stencil_mask(
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
	float2 *sample_position,

	int * o_frag_data,
	int4 * o_frag_stencil_mask
	) {

	int thread_index = GET_ID();

	int ti = thread_index;
	if (ti >= n_fragments) { return; }

	int2 cid_t0 = ts[ti];
	int2 cid_t1 = (ti + 1) != n_fragments ? ts[ti + 1] : make_int2(-1, 0x3f800000);

	int int_t0 = cid_t0.y;
	int int_t1 = cid_t1.y;

	auto intersection_flag_0 = int_t0 & 1;
	auto intersection_flag_1 = int_t1 & 1;

	float t0 = __int_as_float(cid_t0.y & 0xFFFFFFFC);
	float t1 = __int_as_float(cid_t1.y & 0xFFFFFFFC);

	if (t0 < 0) { t0 = 0.f; }
	if (t1 < 0) { t1 = 0.f; }

	if (t0 == 0.f || t0 == 1.f) { intersection_flag_0 = 1; }
	if (t1 == 0.f || t1 == 1.f) { intersection_flag_1 = 1; }

	int cid0 = cid_t0.x;
	int cid1 = cid_t1.x;
	int cid = cid0;
	int pathid = i_curve_path_id[cid0];
	int yx = 0xFFFEFFFE;

	if (cid0 != cid1) { t1 = 1.f; int_t1 = __float_as_int(t1); }
	
	int winding_number_change = 0;
	int scan_winding_number = 0;

	// gen this fragment
	if (t0 < t1) {

		// fetch curve
		uint8_t rev = i_curve_reversed[cid];
		auto curve_type = i_curve_type[cid];
		auto curve_vertex_pos = i_curve_vertex_pos[cid];
		auto p0 = curve_vertex_pos;

		float2 cv0, cv1, cv2, cv3;
		float cw;

		switch (curve_type) {
		default:
			ASSERT(0);
			return;
		case CT_Linear:
			cv0 = vs[p0 + 0];
			cv1 = vs[p0 + 1];
			break;
		case CT_Quadratic:
			cv0 = vs[p0 + 0];
			cv1 = vs[p0 + 1];
			cv2 = vs[p0 + 2];
			break;
		case CT_Cubic:
			cv0 = vs[p0 + 0];
			cv1 = vs[p0 + 1];
			cv2 = vs[p0 + 2];
			cv3 = vs[p0 + 3];
			break;
		case CT_Rational:
			cw = i_curve_arc_w[cid0];
			cv0 = vs[p0 + 0];
			cv1 = vs[p0 + 1];
			cv2 = vs[p0 + 2];
			cv1 *= cw;
			break;
		}

		// subdiv
		#ifndef FRAG_MASK_SUBDIV
		#define FRAG_MASK_SUBDIV 1
		#endif
		float2 curve_p[FRAG_MASK_SUBDIV + 1];
		#pragma unroll
		for (int i = 0; i <= FRAG_MASK_SUBDIV; ++i) {

			float fi = i / (float)FRAG_MASK_SUBDIV;
			float t = (1 - fi) *t0 + fi * t1;

			if (curve_type == CT_Linear) {
				curve_p[i] = d_curve_point<CT_Linear>(cv0, cv1, cv2, cv3, cw, t);
			}
			else if (curve_type == CT_Quadratic) {
				curve_p[i] = d_curve_point<CT_Quadratic>(cv0, cv1, cv2, cv3, cw, t);
			}
			else if (curve_type == CT_Cubic) {
				curve_p[i] = d_curve_point<CT_Cubic>(cv0, cv1, cv2, cv3, cw, t);
			}
			else {
				curve_p[i] = d_curve_point<CT_Rational>(cv0, cv1, cv2, cv3, cw, t);
			}
		}

		// -------- -------- -------- --------e
		float2 vf, vl; // first & last vertex
		int sf, sl;    // first & last side flags

		if (rev) {
			vf = curve_p[FRAG_MASK_SUBDIV];
			vl = curve_p[0];
			sf = intersection_flag_1;
			sl = intersection_flag_0;
		}
		else {
			vf = curve_p[0];
			vl = curve_p[FRAG_MASK_SUBDIV];
			sf = intersection_flag_0;
			sl = intersection_flag_1;
		}

		// position
		int raw_frag_x = d_float2int_rd<FRAG_SIZE>((vf.x + vl.x)*0.5f);
		int raw_frag_y = d_float2int_rd<FRAG_SIZE>((vf.y + vl.y)*0.5f);

		int pos_x = min(max(raw_frag_x, -FRAG_SIZE), (int)((width & 0xFFFFFFFE) + 2));
		int pos_y = min(max(raw_frag_y, -FRAG_SIZE), (int)((height & 0xFFFFFFFE) + 2));

		int y_shift = (pos_y + 0x7FFF) << 16;
		int x_shift = (pos_x + 0x7FFF) & 0xFFFF;

		if ((uint32_t)raw_frag_y < (uint32_t)height) { yx = y_shift | x_shift; }

		// winding number
		int wn_y = pos_y + 1;

		scan_winding_number = 0;
		if (vf.y == vl.y) {
			scan_winding_number = 0;
		}
		else {
			if (vf.y <= wn_y && wn_y < vl.y) {
				scan_winding_number = -1;
			}
			else if (vl.y <= wn_y && wn_y < vf.y) {
				scan_winding_number = 1;
			}
		}

		// -------- -------- -------- --------
		// gen stencil mask

		if (vf.y < vl.y) {
			winding_number_change = -1;
		}
		else if (vf.y > vl.y) {
			winding_number_change = 1;
		}
		else {
			if (vf.x < vl.x) {
				winding_number_change = 1;
			}
			else {
				winding_number_change = -1;
			}
		}

		// -------- -------- -------- --------
		// handle left intersection

		float2 vleft, vright;
		int left_flag = 1;

		if (vf.x < vl.x) {
			vleft = vf;
			vright = vl;
			left_flag = sf;
		}
		else if (vf.x > vl.x) {
			vleft = vl;
			vright = vf;
			left_flag = sl;
		}
		else {

			if (vf.x < pos_x + 1) {

				if (sf == 0) {
					vleft = vf;
					vright = vl;
					left_flag = 0;
				}
				else if (sl == 0) {
					vleft = vl;
					vright = vf;
					left_flag = 0;
				}

			}
		}

//#define USE_IMPLICIT_TEST
#ifdef USE_IMPLICIT_TEST
#else

		int4 lm; // left boundary mask

		if (left_flag == 0) {
			lm = QM_Mask_2x2::fetchVerticalTable<N_SAMPLES>((vleft.y - pos_y) * .5f);

			if (vleft.y <= pos_y + 1) {
				if (vleft.y >= vright.y) { winding_number_change *= -1; }
			}
			else {
				if (vleft.y < vright.y) { winding_number_change *= -1; }
				lm.x = ~lm.x;
				lm.y = ~lm.y;
				lm.z = ~lm.z;
				lm.w = ~lm.w;
			}
		}
		else {
			lm = make_int4(0, 0, 0, 0);
		}

		float2 base = make_float2(pos_x, pos_y);
		#pragma unroll 
		for (int i = 0; i <= FRAG_MASK_SUBDIV; ++i) {
			curve_p[i] = (curve_p[i] - base) * .5f;
		}

		int4 pm = make_int4(0,0,0,0);

		#pragma unroll
		for (int i = 0; i < FRAG_MASK_SUBDIV; ++i) {
			float2 fv1 = curve_p[i];
			float2 fv2 = curve_p[i + 1];

			int4 vm = QM_Mask_2x2::fetchVerticalTable2<N_SAMPLES>(fv1.y, fv2.y);
			int4 tm = QM_Mask_2x2::fetchPixelTable<N_SAMPLES>(fv1, fv2);

			pm.x |= tm.x & vm.x;
			pm.y |= tm.y & vm.y;
			pm.z |= tm.z & vm.z;
			pm.w |= tm.w & vm.w;
		}

		uint32_t mask[4];

		mask[0] = pm.x ^ lm.x;
		mask[1] = pm.y ^ lm.y;
		mask[2] = pm.z ^ lm.z;
		mask[3] = pm.w ^ lm.w;
#endif

		// -------- -------- -------- --------
		o_frag_stencil_mask[thread_index] = make_int4(mask[0], mask[1], mask[2], mask[3]);

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
	| sf * 8  | -     | 
	| sf * 9  | -     | -
	| sf * 10 | -     | scan segment
	*/

	auto o_pos = o_frag_data;
	auto o_sort_index = o_frag_data + stride_fragments;
	auto o_path_id = o_frag_data + stride_fragments * 2;
	int8_t *o_winding_number = (int8_t*)(o_frag_data + stride_fragments * 4);

	auto o_sort_segment = o_frag_data + stride_fragments * 3;
	
	o_pos[ti] = yx;
	o_sort_index[ti] = ti;

	o_path_id[ti] = pathid | winding_number_change << 30;

	o_winding_number[ti] = scan_winding_number;

	// sort segment.
	//int scanseg = 0;
	if (cid0 != cid1) {
		int pathid1 = 0;
		if (cid1 >= 0) { pathid1 = i_curve_path_id[cid1]; }
		else { pathid1 = n_paths; }
		//scanseg = (pathid < pathid1);
		for (int j = pathid + 1; j <= pathid1; j++) {
			o_sort_segment[j] = ti + 1;
		}
	}
	else {
		//scanseg = 0;
	}
	if (!ti) {
		o_pos[n_fragments] = -1;
		for (int j = 0; j <= pathid; j++) { o_sort_segment[j] = 0; }
	}

}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_shuffleFragments(
	int* pfragi, int stride_fragments, int n_fragments) {

	int i = GET_ID();
	if (i >= n_fragments) { return; }

	// 4 to 3
	int idx = pfragi[i + stride_fragments];

	int8_t *wn_in = (int8_t*)(pfragi + stride_fragments * 4);
	int8_t *wn_out = (int8_t*)(pfragi + stride_fragments * 3);
	wn_out[i] = wn_in[idx];
}

// -------- -------- -------- -------- -------- -------- -------- --------
namespace SingleDraw {

// -------- -------- -------- -------- -------- -------- -------- --------
template <int FRAG_SIZE>
__global__ void k_mark_merged_fragment_and_span(
	uint8_t *pathFillRule,
	int32_t *pfrag,
	int32_t nfrag,
	int32_t stride_fragments,
	int width, int height
	) {

	int i = GET_ID();
	if (i >= nfrag) { return; }

	uint32_t *o_path_frag_flag = (uint32_t*)pfrag + 4 * stride_fragments;
	uint32_t *o_span_flag = o_path_frag_flag + nfrag;
	uint32_t path_frag_flag=0;
	uint32_t span_flag=0;
	
	uint32_t *i_yx = (uint32_t*)pfrag;
	uint32_t yx_1 = i_yx[i];

	int16_t x1 = ((int16_t)(yx_1 & 0xFFFF)) - 0x7FFF;
	int16_t y1 = ((int16_t)(yx_1 >> 16)) - 0x7FFF;

	// out-of-boundary fragment *may* generage span 
	// but *never* generates merged fragment

	if (i == 0) {
		if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height) {
			path_frag_flag = 0;
		}
		else {
			path_frag_flag = 1;
		}
		span_flag = 0;
	}
	else {

		uint32_t *i_pid = (uint32_t*)pfrag + 2 * stride_fragments;
		int8_t *i_winding_number = (int8_t*)((int32_t*)pfrag + 3 * stride_fragments);

		uint32_t pid_0 = i_pid[i - 1] & 0x3FFFFFFF;
		uint32_t pid_1 = i_pid[i] & 0x3FFFFFFF;

		uint8_t fill_rule = pathFillRule[pid_1];

		uint32_t yx_0 = i_yx[i - 1];

		int16_t x0 = ((int16_t)(yx_0 & 0xFFFF)) - 0x7FFF;
		int16_t y0 = ((int16_t)(yx_0 >> 16)) - 0x7FFF;

		if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height) {
			path_frag_flag = 0;
		}
		else if (pid_0 != pid_1 || yx_0 != yx_1) {
			path_frag_flag = 1;
		}
		else {
			path_frag_flag = 0;
		}

		int8_t wn = i_winding_number[i];
		bool wn_flag = ((fill_rule == 0) && (wn != 0)) || ((fill_rule == 1) && (wn & 1));
		
		if (y0 == y1 && ((x0 + FRAG_SIZE) < x1) && pid_0 == pid_1 && wn_flag) {
			span_flag = 1;
		}
		else {
			span_flag = 0;
		}
	}

	o_path_frag_flag[i]=path_frag_flag;
	o_span_flag[i]=span_flag;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template <class T>
__forceinline__ __device__ void d_write_mask(T *p, int4 mask) {
	*((int*)0) = 0;
}

template <>
__forceinline__ __device__ void d_write_mask<int>(int *p, int4 mask) {
	*p = mask.w;
}

template <>
__forceinline__ __device__ void d_write_mask<int4>(int4 *p, int4 mask) {
	*p = mask;
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<class MASK_TYPE, int N_SAMPLES,int FRAG_SIZE, int BLOCK_SIZE>
__global__ void k_gen_stencil_cuda_sm(
	uint8_t *i_pathFillRule,
	int32_t *i_pfrag,
	uint32_t nfrag,
	uint32_t sn,
	int4* i_masks,
	int2 *o_block_boundary_bins,
	MASK_TYPE *o_stencil,
	int width, int height
	) {
	////////////////////
	//init the bins, use int7x4 hacky SIMD
	//note the weird packing order

	__shared__ int32_t bins[BLOCK_SIZE * 4];
	////////////////////
	int i = GET_ID();
	int sm_idx = threadIdx.x;
	int acc_idx = sm_idx;
	int4 mask_i = make_int4(0, 0, 0, 0);
	int fill_rule = 1; //even-odd is faster
	MASK_TYPE* o_stencil_i = NULL;
	int2* o_bins_i = NULL;

	int32_t init_pos = 0;
	int32_t init_neg = 0;

	bool out_of_boundary;
	{
		int32_t pos = i_pfrag[i];
		int16_t rc_x = ((int16_t)(pos & 0xFFFF)) - 0x7FFF;
		int16_t rc_y = ((int16_t)(pos >> 16)) - 0x7FFF;
		out_of_boundary = (rc_x < 0 || rc_y < 0 || rc_x >= width || rc_y >= height);
	}
	
	if (i < nfrag && i_pfrag[i] != 0xFFFEFFFE && !out_of_boundary) {

		auto pid = i_pfrag[i + sn * 2] & 0x3FFFFFFF;
		fill_rule = i_pathFillRule[pid];

		/////////
		int raw_index = i_pfrag[i + sn];

		// sn * 4: flag
		// sn * 6: index, inclusive_scan
		int fragIndex_block_min = i_pfrag[blockIdx.x*BLOCK_SIZE + sn * 6] - 1;
		int i_block_max = blockIdx.x*BLOCK_SIZE + (BLOCK_SIZE - 1);
		int fragIndex_block_max = i_block_max < nfrag ? i_pfrag[i_block_max + sn * 6] - 1 : -1;

		//
		int fragIndex = i_pfrag[i + sn * 6] - 1;

		sm_idx = fragIndex - fragIndex_block_min;
		acc_idx = sm_idx;
		mask_i = i_masks[raw_index];
		int winding_number_change = i_pfrag[raw_index + sn * 2] >> 30;
		if (winding_number_change == 0) {
			mask_i = make_int4(0, 0, 0, 0);
		}
		else if (winding_number_change < 0) {
			acc_idx += BLOCK_SIZE;
		}

		if (threadIdx.x == 0) {
			//we need to write the boundary bins
			//shift by 1 for aligned fetch in the boundary pass
			o_bins_i = o_block_boundary_bins + 1 + (blockIdx.x * 2);
		}
		//it's the start of a new fragment, we need to write out a final stencil
		//this is true even if we're the head/tail of a block
		//if the tail goes into the next block, we overwrite it in the next pass
		// and, we need add fragment init winding number.
		if (!i || fragIndex != (i_pfrag[i - 1 + sn * 6] - 1)) {
			o_stencil_i = o_stencil + fragIndex;
			int init_winding_number = ((int8_t *)(i_pfrag + sn * 3))[i];
			if (init_winding_number < 0) {
				init_neg = -0x01010101 * init_winding_number;
			}
			else {
				init_pos = 0x01010101 * init_winding_number;
			}
			if (fragIndex == fragIndex_block_max) {
				//use this thread to avoid a race condition later
				o_bins_i = o_block_boundary_bins + 2 + (blockIdx.x * 2);
			}
		}

	}

	//init
	if (o_stencil_i || !threadIdx.x) {
		bins[sm_idx + BLOCK_SIZE * 0] = init_pos;
		bins[sm_idx + BLOCK_SIZE * 1] = init_neg;
		bins[sm_idx + BLOCK_SIZE * 2] = init_pos;
		bins[sm_idx + BLOCK_SIZE * 3] = init_neg;
	}
	__syncthreads();

	if (out_of_boundary) {
#pragma unroll
		for (int pixel_id = 0; pixel_id < (N_SAMPLES == 8 ? 1 : FRAG_SIZE*FRAG_SIZE); pixel_id++) {
#pragma unroll
			for (int bit_id = 0; bit_id < 8; bit_id += 2) {
				__syncthreads();
				__syncthreads();
			}
		}
	}
	else {

#pragma unroll
		for (int pixel_id = 0; pixel_id < (N_SAMPLES == 8 ? 1 : FRAG_SIZE*FRAG_SIZE); pixel_id++) {
			int32_t ret_stencil = 0;
#pragma unroll
			for (int bit_id = 0; bit_id < 8; bit_id += 2) {
				//QM: the !threadIdx.x could be causing a race condition if there is an even-odd fast path
				//accumulate
				int acc0 = ((mask_i.x >> bit_id) & 0x01010101);
				int acc1 = ((mask_i.x >> (bit_id + 1)) & 0x01010101);
				atomicAdd((int*)bins + acc_idx, acc0);
				atomicAdd((int*)bins + acc_idx + BLOCK_SIZE * 2, acc1);
				//atomicAdd((uint64_t*)bins + acc_idx, (uint64_t)(uint32_t(acc0))+((uint64_t)(uint32_t(acc1))<<32));
				__syncthreads();
				//fetch the result and test the fill rule
				int32_t winding_number_batch0 = ((0x80808080 + bins[sm_idx + BLOCK_SIZE * 0] - bins[sm_idx + BLOCK_SIZE * 1]) & 0x7f7f7f7f);
				int32_t winding_number_batch1 = ((0x80808080 + bins[sm_idx + BLOCK_SIZE * 2] - bins[sm_idx + BLOCK_SIZE * 3]) & 0x7f7f7f7f);
				if (o_bins_i) {
					//write the head/tail bins
					o_bins_i[0] = make_int2(winding_number_batch0, winding_number_batch1);
					o_bins_i += gridDim.x * 2;
				}
				if (o_stencil_i || !threadIdx.x) {
					////////////
					//it's fine to over-do this for thread 0
					if (fill_rule == 0) {
						//non-zero mode, propagate the whole winding number to the lowest bit
						winding_number_batch0 |= ((winding_number_batch0 >> 4));
						winding_number_batch0 |= ((winding_number_batch0 >> 2));
						winding_number_batch0 |= ((winding_number_batch0 >> 1));
						winding_number_batch1 |= ((winding_number_batch1 >> 4));
						winding_number_batch1 |= ((winding_number_batch1 >> 2));
						winding_number_batch1 |= ((winding_number_batch1 >> 1));
					}
					else {
						//even-odd mode, do nothing
					}
					ret_stencil |= (winding_number_batch0 & 0x01010101) << bit_id;
					ret_stencil |= (winding_number_batch1 & 0x01010101) << (bit_id + 1);
					////////////
					//re-init
					//gotta init the accumulator for thread 0, even if it doesn't end up writing anything
					bins[sm_idx + BLOCK_SIZE * 0] = init_pos;
					bins[sm_idx + BLOCK_SIZE * 1] = init_neg;
					bins[sm_idx + BLOCK_SIZE * 2] = init_pos;
					bins[sm_idx + BLOCK_SIZE * 3] = init_neg;
				}
				__syncthreads();
			}
			//shift ret_stencil into mask_i for the mid-block final output
			mask_i.x = mask_i.y; mask_i.y = mask_i.z; mask_i.z = mask_i.w;
			mask_i.w = ret_stencil;
		}

	}

	if (o_stencil_i) { d_write_mask<MASK_TYPE>(o_stencil_i, mask_i); }

}

template<class MASK_TYPE, int N_SAMPLES,int FRAG_SIZE, int PREV_BLOCK_SIZE>
__global__ void k_gen_stencil_block_boundary(
	uint8_t *i_pathFillRule,
	int32_t *i_pfrag,
	uint32_t sn,
	int2 *i_block_boundary_bins,
	int n_blocks,
	MASK_TYPE *o_stencil,
	int4* i_masks,
	int width, int height
	) {
	//only write out if the ids are the same
	int i = GET_ID() + 1;
	if(i>=n_blocks){return;}
	int fragIndex0 = i_pfrag[i*PREV_BLOCK_SIZE-1 + sn * 6] - 1;
	int fragIndex1 = i_pfrag[i*PREV_BLOCK_SIZE + sn * 6] - 1;
	if (fragIndex0 != fragIndex1
		|| i_pfrag[i*PREV_BLOCK_SIZE - 1] == 0xFFFEFFFE
		|| i_pfrag[i*PREV_BLOCK_SIZE] == 0xFFFEFFFE) {
		return;
	}
	auto pid = i_pfrag[i*PREV_BLOCK_SIZE + sn * 2] & 0x3FFFFFFF;
	
	int fill_rule = i_pathFillRule[pid];
	int4* i_bins=(int4*)i_block_boundary_bins+i;
	int4 mask_i=make_int4(0,0,0,0);

	{
		for(int pixel_id=0;pixel_id<(N_SAMPLES==8?1:FRAG_SIZE*FRAG_SIZE);pixel_id++){
			int32_t ret_stencil=0;
			for(int bit_id=0;bit_id<8;bit_id+=2){
				int4 i_bins_cur=((int4*)i_block_boundary_bins)[i+n_blocks*(pixel_id*4+(bit_id>>1))];
				//nvcc bug: i_bins[0] doesn't work
				//int4 i_bins_cur=i_bins[0];
				int32_t winding_number_batch0 = ((i_bins_cur.x + i_bins_cur.z) & 0x7f7f7f7f);
				int32_t winding_number_batch1 = ((i_bins_cur.y + i_bins_cur.w) & 0x7f7f7f7f);
				////////////////
				if(fill_rule==0){
					winding_number_batch0 |= ((winding_number_batch0 >> 4));
					winding_number_batch0 |= ((winding_number_batch0 >> 2));
					winding_number_batch0 |= ((winding_number_batch0 >> 1));
					winding_number_batch1 |= ((winding_number_batch1 >> 4));
					winding_number_batch1 |= ((winding_number_batch1 >> 2));
					winding_number_batch1 |= ((winding_number_batch1 >> 1));
				}
				ret_stencil|=(winding_number_batch0&0x01010101)<<bit_id;
				ret_stencil|=(winding_number_batch1&0x01010101)<<(bit_id+1);
				//if(p0==P0_DUMP){
				//	printf("%08x\n",ret_stencil);
				//}
				i_bins+=n_blocks;
			}
			mask_i.x=mask_i.y;mask_i.y=mask_i.z;mask_i.z=mask_i.w;
			mask_i.w=ret_stencil;
		}

	}

	//
	if (0 <= fragIndex0) {
		d_write_mask<MASK_TYPE>(o_stencil + fragIndex0, mask_i);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
template <int FRAG_SIZE>
__global__ void k_gen_gl_merged_fragment_and_span(

	int32_t *pfrag,

	const uint32_t * __restrict__ pathFillInfo,

	int4 *o_gl_draw_data,

	int32_t nfrag,
	uint32_t sn,

	int32_t n_merged_fragments,
	int32_t n_spans,

	int width, int height

	) {

	int i = GET_ID();
	if (i >= nfrag) { return; }

	int frag_flag = pfrag[i + sn * 4];
	int frag_index = pfrag[i + sn * 6];

	int span_flag = pfrag[i + sn * 4 + nfrag];
	int span_index = pfrag[i + sn * 6 + nfrag] - n_merged_fragments;

	// check
	//
	// f-flag: 0 1 0 1 0 0 0
	// s-flag: 0 0 0 1 0 1 0
	//
	// f-pos:  0 1 1 2 2 2 2 
	// s-pos:  0 0 0 1 1 2 2

	int32_t num_of_frag_before = frag_index - frag_flag;
	int32_t num_of_span_before = span_index - span_flag;

	if (frag_flag) {
		int output_index = num_of_frag_before + num_of_span_before;

		int32_t raw_pos = pfrag[i];
		int16_t rc_x = ((int16_t)(raw_pos & 0xFFFF)) - 0x7FFF;
		int16_t rc_y = ((int16_t)(raw_pos >> 16)) - 0x7FFF;

		auto pid = pfrag[i + sn * 2] & 0x3FFFFFFF;
		auto fillInfo = pathFillInfo[pid]; // path fill info

		int32_t pos_yx = (((int32_t)rc_y) << 16) | rc_x;

		o_gl_draw_data[output_index] = make_int4(pos_yx, 2, fillInfo, frag_index);
		
	}
	if (span_flag) {

		int output_index = num_of_frag_before + num_of_span_before + frag_flag;

		uint32_t p0 = pfrag[i-1];
		uint32_t p1 = pfrag[i];

		uint32_t pid = pfrag[i + sn * 2] & 0x3FFFFFFF;

		int16_t y0 = ((int16_t)(p0 >> 16)) - 0x7FFF;

		int16_t x0 = ((int16_t)(p0 & 0xFFFF)) - 0x7FFF + FRAG_SIZE;
		int16_t x1 = ((int16_t)(p1 & 0xFFFF)) - 0x7FFF;
		x0 = max((int16_t)0, x0);

		// output:
		//   int2 | pos (x,y)
		//   uint32 | width
		//   uint32 | rgba or gradient

		uint32_t fillInfo = pathFillInfo[pid];
		o_gl_draw_data[output_index] = make_int4((((int32_t)y0) << 16) | x0, x1 - x0, fillInfo, 0);
		
	}

}

}

extern int tiger_transform_x;
extern int tiger_transform_y;

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// @function rasterizeImpl
//
template <int FRAG_SIZE>
void VGRasterizer::rasterizeImpl() {

	static bool _first = true;
	if (_first) { _first = false; _verbose = true; }
	else { _verbose = false; }

	if (_verbose) { printf("--- rasterize\n"); }

	if (_enable_step_timing) {
		_step_timestamp.clear();
		long long ts;
		QueryPerformanceCounter((LARGE_INTEGER*)&ts);
		_step_timestamp.push_back(ts);
	}

	// -------- -------- -------- --------
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("beginning");

	g_alloc.reset();

	int n_paths;
	int n_curves;
	int n_vertices;

	if (_animation) {
		static int start_time = -1;
		static int last_frame_time_stamp = 0;
		int atime = 0;
		if (start_time == -1) { start_time = clock(); }
		atime = clock() - start_time;

		try {
			vg_animation(last_frame_time_stamp, atime,
				_vg_in_curve[0], _vg_in_curve[1],
				_vg_in_path[0], _vg_in_path[1]);
			_current_input_pack_id = 1;

		}
		catch (...) {
		}

		last_frame_time_stamp = atime;
	}

	auto &_in = _animation ? _vg_in_curve[_current_input_pack_id] : _vg_in_curve[1];
	auto &_path_in = _animation ? _vg_in_path[_current_input_pack_id] : _vg_in_path[0];

	n_paths = _path_in.n_paths;
	n_curves = _in.n_curves;
	n_vertices = _in.n_vertices;

	_element_number.path = n_paths;
	_element_number.curve = n_curves;
	_element_number.vertex = n_vertices;

	if (_gpu_stroke_to_fill && _draw_curve) {
		auto &_fill_in = _vg_in_curve[1];
		_fill_in.vertex.resizeWithoutCopy(_gpu.transformedVertex.size());
		cudaMemcpy(_fill_in.vertex.gptr(), _gpu.transformedVertex.gptr(), _gpu.transformedVertex.size() * sizeof(float2),
			cudaMemcpyDeviceToDevice);
	}

	if(!_gpu_stroke_to_fill) {

		_gpu.transformedVertex.resizeWithoutCopy(n_vertices);
		_base_gpu_is_path_visible.resizeWithoutCopy(n_paths);
		cudaMemsetAsync(_base_gpu_is_path_visible.gptr(), 0, n_paths * sizeof(uint64_t));

		float z = (float)std::min(_output_buffer_size.x, _output_buffer_size.y);
		LAUNCH(k_transformVertex, n_vertices, 256, (
			_in.n_vertices, _in.vertex, _in.vertex_path_id,
			_input_transform[0], _input_transform[1], _input_transform[2], _input_transform[3],
			(float)_output_buffer_size.x, (float)_output_buffer_size.y, z, -z * 0.5f,
			_gpu.transformedVertex,
			(char*)_base_gpu_is_path_visible.gptr()
			));
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("resize vertex & path visible");
	}

	// -------- -------- -------- --------
	// -------- -------- -------- --------

	_gpu.curve_pixel_count.resizeWithoutCopy(n_curves + 1);
	_gpu.monotonic_cutpoint_cache.resizeWithoutCopy(n_curves * 5);

	//
	LAUNCH((k_make_intersection_0<FRAG_SIZE, 256>), n_curves, 256, (
		nullptr,
		(float4*)_gpu.monotonic_cutpoint_cache.gptr(),
		(char*)((int*)_gpu.monotonic_cutpoint_cache.gptr() + n_curves * 4),
		_gpu.curve_pixel_count,
		nullptr, _in.curve_type, _in.curve_vertex_pos,
		n_curves, 0,
		_gpu.transformedVertex, _in.curve_arc_w, _in.curve_path_id,
		_output_buffer_size.x, _output_buffer_size.y,
		nullptr, 
		_base_gpu_is_path_visible.gptr()
		));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("make_intersection 0");

	int n_fragments = escan((int*)_gpu.curve_pixel_count.gptr(), n_curves);	
	_element_number.curve_fragment = n_fragments;

	if (!n_fragments) { return; }
	int stride_fragments = (n_fragments + 256)&-256;

	if (_verbose) {
		printf(">>> n_curves %d\n", n_curves);
		printf(">>> n_fragments %d\n", n_fragments);
		printf(">>> stride_fragments %d\n", stride_fragments);
	}

	//
	_gpu.intersection.resizeWithoutCopy(n_fragments * 2 + 2);
	_gpu.fragmentData.resizeWithoutCopy(8 * stride_fragments + 1);

	cuglUpdateBuffer(sizeof(int) * 4 * n_fragments,
		_gl.buffer.stencilDrawMask, _cuda.resource.stencilDrawMask);
	int4 *p_stencil_draw_mask = (int4 *)cuglMap(_cuda.resource.stencilDrawMask);

	LAUNCH((k_make_intersection_1<FRAG_SIZE, 256>), n_curves, 256, (
		(int2*)_gpu.intersection.gptr(),
		(float4*)_gpu.monotonic_cutpoint_cache.gptr(),
		(char*)((int*)_gpu.monotonic_cutpoint_cache.gptr() + n_curves * 4),
		(int*)_gpu.curve_pixel_count.gptr(),
		nullptr, _in.curve_type, _in.curve_vertex_pos,
		n_curves, n_fragments,
		(float2*)_gpu.transformedVertex.gptr(), _in.curve_arc_w, _in.curve_path_id,
		_output_buffer_size.x, _output_buffer_size.y,
		nullptr, 
		_base_gpu_is_path_visible.gptr()));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("make_intersection 1");

	if(_samples==8){
		LAUNCH((k_gen_fragment_and_stencil_mask<8, FRAG_SIZE, 256>), n_fragments, 256, (
			n_paths, n_curves, n_fragments, stride_fragments,
			(int2*)_gpu.intersection.gptr(),
			_in.curve_path_id, _in.curve_vertex_pos, _in.curve_type, _in.curve_reversed,
			_in.curve_arc_w, _gpu.transformedVertex,
			_output_buffer_size.x, _output_buffer_size.y,
			_sample_position.gptr(),			
			_gpu.fragmentData, p_stencil_draw_mask
			));
	}else{
		LAUNCH((k_gen_fragment_and_stencil_mask<32,FRAG_SIZE, 256>), n_fragments, 256, (
			n_paths, n_curves, n_fragments, stride_fragments,
			(int2*)_gpu.intersection.gptr(),
			_in.curve_path_id, _in.curve_vertex_pos, _in.curve_type, _in.curve_reversed,
			_in.curve_arc_w, _gpu.transformedVertex,
			_output_buffer_size.x, _output_buffer_size.y,
			_sample_position.gptr(),
			_gpu.fragmentData, p_stencil_draw_mask
			));
	}
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("gen fragments");

	/*
	after gen_fragment_and_stencil_mask
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
	*/

	/////////////////////////////
	//sort by the yx component, per-path segments... generate earlier
	cuda_seg_sort_int_by_int(
		_gpu.fragmentData.gptr(),
		_gpu.fragmentData.gptr() + stride_fragments,
		n_fragments,
		_gpu.fragmentData.gptr() + stride_fragments * 3,
		n_paths);
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("sort fragments");

	/*
	after sort
	----------------------------------------------------------------
	| offset  | size  |
	| sf * 0  | nf    | position (sorted)
	| sf * 1  | nf    | index (sorted)
	| sf * 2  | nf    | pathid & winding_number_change
	| sf * 3  | -     | -
	| sf * 4  | nf    | winding number
	| sf * 5  | -     | -
	| sf * 6  | -     | -
	| sf * 7  | -     | -
	*/

	LAUNCH(k_shuffleFragments, n_fragments, 256, (_gpu.fragmentData.gptr(),
		stride_fragments, n_fragments));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("shuffle");

	if (_enable_step_timing) {
		cudaDeviceSynchronize();
		long long ts;
		QueryPerformanceCounter((LARGE_INTEGER*)&ts);
		_step_timestamp.push_back(ts);
	}

	/*
	after shuffle
	----------------------------------------------------------------
	| offset  | size  |
	| sf * 0  | nf    | position (sorted)
	| sf * 1  | nf    | index (sorted)
	| sf * 2  | nf    | pathid & winding_number_change
	| sf * 3  | nf    | winding number (sorted)
	| sf * 4  | -     | 
	| sf * 5  | -     | -
	| sf * 6  | -     | -
	| sf * 7  | -     | -
	*/

	thrust_exclusive_scan(
		(int8_t*)(_gpu.fragmentData.gptr() + stride_fragments * 3),n_fragments,
		(int8_t*)(_gpu.fragmentData.gptr() + stride_fragments * 3));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("scan abuf");

	/*
	after scan winding number
	----------------------------------------------------------------
	| offset  | size  |
	| sf * 0  | nf    | position (sorted)
	| sf * 1  | nf    | index (sorted)
	| sf * 2  | nf    | pathid & winding_number_change
	| sf * 3  | nf    | winding number (sorted & scaned)
	| sf * 4  | -     |
	| sf * 5  | -     | -
	| sf * 6  | -     | -
	| sf * 7  | -     | -
	*/

	LAUNCH(SingleDraw::k_mark_merged_fragment_and_span<FRAG_SIZE>, n_fragments, 256, (
		_path_in.fill_rule.gptr(),
		_gpu.fragmentData.gptr(),
		n_fragments, stride_fragments,
		_output_buffer_size.x, _output_buffer_size.y
		));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("after countPathFragment");

	/*
	----------------------------------------------------------------
	| offset  | size  |
	| sf * 0  | nf    | position (sorted)
	| sf * 1  | nf    | index (sorted)
	| sf * 2  | nf    | pathid & winding_number_change
	| sf * 3  | nf    | winding number (sorted & scaned)
	| sf * 4  | 2*nf  | merged fragment flag, span flag
	| sf * 5  | *     |
	| sf * 6  | -     | -
	| sf * 7  | -     | -
	*/

	thrust_inclusive_scan(
		(uint32_t*)_gpu.fragmentData.gptr() + stride_fragments * 4,
		n_fragments * 2,
		(uint32_t*)_gpu.fragmentData.gptr() + stride_fragments * 6
		);
	int n_output_fragments;
	int n_spans;
	int scan_result[2];
	cudaMemcpy2D(scan_result, sizeof(int),
		_gpu.fragmentData.gptr() + stride_fragments * 6 + n_fragments - 1,
		n_fragments * sizeof(int), sizeof(int), 2, cudaMemcpyDeviceToHost);

	n_output_fragments = scan_result[0];
	n_spans = scan_result[1];

	n_spans -= n_output_fragments;

	_element_number.merged_fragment = n_output_fragments;
	_element_number.span = n_spans;

	/*
	----------------------------------------------------------------
	| offset  | size  |
	| sf * 0  | nf    | position (sorted)
	| sf * 1  | nf    | index (sorted)
	| sf * 2  | nf    | pathid & winding_number_change
	| sf * 3  | nf    | winding number (sorted & scaned)
	| sf * 4  | 2*nf  | merged fragment flag, span flag
	| sf * 5  | *     |
	| sf * 6  | 2*nf  | merged fragment index, span index
	| sf * 7  | *     |
	*/

	// -------- -------- -------- --------
	if (_verbose) {
		printf("n_output_fragments: %d\n", n_output_fragments);
		printf("n_spans: %d\n", n_spans);
	}

	// -------- -------- -------- --------
	// update buffer size
	cuglUpdateBuffer(sizeof(int4) * (n_output_fragments + n_spans), _gl.buffer.outputIndex,
		_cuda.resource.outputIndex);
	cuglUpdateBuffer(sizeof(int4) * n_fragments, _gl.buffer.qm_output_stencil_mask,
		_cuda.resource.qm_output_stencil_mask);

	// -------- -------- -------- --------
	// map buffer pointer.
	int4 * p_output_index = (int4*)cuglMap(_cuda.resource.outputIndex);
	int4* p_qm_output_sample_mask = (int4*)cuglMap(_cuda.resource.qm_output_stencil_mask);

	// -------- -------- -------- -------
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("before k_gen_stencil_block_boundary");

	//#define GEN_STENCIL_BLOCK_SIZE 256
#define GEN_STENCIL_BLOCK_SIZE 1024
	int n_blocks_gen_stencil = divup(n_fragments, GEN_STENCIL_BLOCK_SIZE);
	if (_samples == 8) {
		_gpu.blockBoundaryBins.resizeWithoutCopy(FRAG_SIZE*FRAG_SIZE * 2 * 2 * n_blocks_gen_stencil + 2);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		LAUNCH((SingleDraw::k_gen_stencil_cuda_sm<int, 8, FRAG_SIZE, GEN_STENCIL_BLOCK_SIZE>), \
			n_fragments, GEN_STENCIL_BLOCK_SIZE, (
				_path_in.fill_rule.gptr(),
				_gpu.fragmentData, n_fragments, stride_fragments,
				(int4*)p_stencil_draw_mask,
				(int2*)_gpu.blockBoundaryBins.gptr(),
				(int*)p_qm_output_sample_mask,
				_output_buffer_size.x, _output_buffer_size.y
				));
		cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_stencil_cuda_sm");
		if (n_blocks_gen_stencil > 1) {
			LAUNCH((SingleDraw::k_gen_stencil_block_boundary<int, 8, FRAG_SIZE, GEN_STENCIL_BLOCK_SIZE>), \
				n_blocks_gen_stencil - 1, 256, (
					_path_in.fill_rule.gptr(),
					_gpu.fragmentData, stride_fragments,
					(int2*)_gpu.blockBoundaryBins.gptr(),
					n_blocks_gen_stencil,
					(int*)p_qm_output_sample_mask,
					//todo
					(int4*)p_stencil_draw_mask,
					_output_buffer_size.x, _output_buffer_size.y
					));
			DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_stencil_block_boundary");
		}
	}
	else {
		_gpu.blockBoundaryBins.resizeWithoutCopy(FRAG_SIZE*FRAG_SIZE * 8 * 2 * n_blocks_gen_stencil + 2);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		LAUNCH((SingleDraw::k_gen_stencil_cuda_sm<int4, 32, FRAG_SIZE, GEN_STENCIL_BLOCK_SIZE>), \
			n_fragments, GEN_STENCIL_BLOCK_SIZE, (
				_path_in.fill_rule.gptr(),
				_gpu.fragmentData, n_fragments, stride_fragments,
				(int4*)p_stencil_draw_mask,
				(int2*)_gpu.blockBoundaryBins.gptr(),
				p_qm_output_sample_mask,
				_output_buffer_size.x, _output_buffer_size.y
				));
		cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
		//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_stencil_cuda_sm");
		if (n_blocks_gen_stencil > 1) {
			LAUNCH((SingleDraw::k_gen_stencil_block_boundary<int4, 32, FRAG_SIZE, GEN_STENCIL_BLOCK_SIZE>), \
				n_blocks_gen_stencil - 1, 256, (
					_path_in.fill_rule.gptr(),
					_gpu.fragmentData, stride_fragments,
					(int2*)_gpu.blockBoundaryBins.gptr(),
					n_blocks_gen_stencil,
					p_qm_output_sample_mask,
					//todo
					(int4*)p_stencil_draw_mask,
					_output_buffer_size.x, _output_buffer_size.y
					));
			DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_stencil_block_boundary");
		}
	}
#undef GEN_STENCIL_BLOCK_SIZE

	LAUNCH((SingleDraw::k_gen_gl_merged_fragment_and_span<FRAG_SIZE>), n_fragments, 256, (
		_gpu.fragmentData.gptr(), _path_in.fill_info.gptr(), p_output_index,
		n_fragments, stride_fragments, n_output_fragments, n_spans,
		_output_buffer_size.x, _output_buffer_size.y
		));
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_fragment_and_span");

	if (_count_pixel) {

		auto count_vg2_pixel = [&]() {
			std::vector<int4> cpu_gl_data;
			int32_t n = n_output_fragments + n_spans;
			cpu_gl_data.resize(n);
			cudaMemcpy(cpu_gl_data.data(), p_output_index,
				sizeof(int4) * n, cudaMemcpyDeviceToHost);

			int count = 0;
			for (int i = 0; i < n; ++i) {
				auto e = cpu_gl_data[i];
				auto w = e.y;
				count += FRAG_SIZE * w;
			}
			_pixel_count = count;
		};

		count_vg2_pixel();
	}

	// -------- -------- -------- --------
	cuglUnMap(_cuda.resource.stencilDrawMask);
	cuglUnMap(_cuda.resource.outputIndex);
	cuglUnMap(_cuda.resource.qm_output_stencil_mask);
	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("unmap");

	if (_break_before_gl) {
		cudaDeviceSynchronize();
		return;
	}

	// -------- -------- -------- --------
	// output part
	// -------- -------- -------- --------

	if (_enable_step_timing) {
		cudaDeviceSynchronize();
		long long ts;
		QueryPerformanceCounter((LARGE_INTEGER*)&ts);
		_step_timestamp.push_back(ts);
	}

	if (_samples == 32) {
		_gl.texture.stencilDrawMask.buffer(GL_RGBA32I, _gl.buffer.qm_output_stencil_mask);
	}
	else if (_samples == 8) {
		_gl.texture.stencilDrawMask.buffer(GL_R32I, _gl.buffer.qm_output_stencil_mask);
	}
	else {
		// throw.
	}

	_gl.texture.outputIndex.buffer(GL_RGBA32I, _gl.buffer.outputIndex);

	// -------- -------- -------- --------
	// draw

	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("before GL draw");

	DEBUG_CHECK_GL_ERROR();

	if (!_enable_tiger_clip) {
		glDisable(GL_STENCIL_TEST);
	}
	else {
		glEnable(GL_STENCIL_TEST);
		glClearStencil(0);
		glClearStencil(0);
		glStencilMask(~0);
		glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
		glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
	}
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DEPTH_TEST);

	glLineWidth(2.0f);

	DEBUG_CHECK_GL_ERROR();

	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	if (_multisample_output) {
		glEnable(GL_MULTISAMPLE);
	}
	else {
		glDisable(GL_MULTISAMPLE);
	}

	// -------- -------- -------- 
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// draw to output
	_gl.texture.outputIndex.bindUnit(0);
	_gl.texture.stencilDrawMask.bindUnit(3);

	_base_gl_texture_gradient_table.bindUnit(4);
	_base_gl_texture_gradient_ramp.bindUnit(5);

	uint32_t pathNumberUpperBound = 2;
	uint32_t pathNumber = (uint32_t)_path_in.n_paths;
	while (pathNumberUpperBound < pathNumber) { pathNumberUpperBound <<= 1; }

	glBindFramebuffer(GL_FRAMEBUFFER, _base_gl_framebuffer_output);

	_gl.program.output.uniform3f("pid2depth_irampheight",
		-1.f / (float)pathNumberUpperBound, 1.f, _gradient_irampheight);
	_gl.program.output.uniform3fv("inv_proj_rx", 1, _inv_projection_context + 0);
	_gl.program.output.uniform3fv("inv_proj_ry", 1, _inv_projection_context + 3);
	_gl.program.output.uniform3fv("inv_proj_rw", 1, _inv_projection_context + 6);
	_gl.program.output.uniform3fv("inv_proj_rp", 1, _inv_projection_context + 9);
	_gl.program.output.uniform1f("inv_proj_a", _inv_projection_context[12]);
	_gl.program.output.uniform1i("enable_srgb_correction", _enableSRGBCorrection ? 1 : 0);

	if (_enableSRGBCorrection) {
		glEnable(GL_FRAMEBUFFER_SRGB);
	}
	else {
		glDisable(GL_FRAMEBUFFER_SRGB);
	}

	glViewport(0, 0, _output_buffer_size.x, _output_buffer_size.y);
#ifdef _DEBUG
	//glClearColor(0, 0, 1, 1);
	glClearColor(1, 1, 1, 1);
#else
	if (_animation) {
		glClearColor( 0.2f, 0.2f, 0.2f, 1);
	}
	else {
		glClearColor(1, 1, 1, 1);
	}
#endif

	if (!_enable_tiger_clip) { glClear(GL_COLOR_BUFFER_BIT); }
	else {
		glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		if (!_base_gl_tiger_clip_path) {
			_base_gl_tiger_clip_path = glGenPathsNV(1);
			auto svg = tigerClipString();
			glPathStringNV(_base_gl_tiger_clip_path, GL_PATH_FORMAT_SVG_NV,
				(GLsizei)svg.length(), svg.c_str());
		}

		glUseProgram(0);
		glMatrixLoadIdentityEXT(GL_PROJECTION);
		glMatrixLoadIdentityEXT(GL_MODELVIEW);
		glMatrixScalefEXT(GL_MODELVIEW, 1.2f, 1.2f, 1.2f);
		glMatrixScalefEXT(GL_MODELVIEW, 1.f, -1.f, 1.f);
		glMatrixOrthoEXT(GL_MODELVIEW, 0, 1024, 0, 1024, -1, 1);
		glMatrixTranslatefEXT(GL_MODELVIEW, (float)tiger_transform_x, (float)tiger_transform_y, 0.f);
		float s = 1.6f;
		glMatrixScalefEXT(GL_MODELVIEW, s, s, s);

		glStencilFillPathNV(_base_gl_tiger_clip_path, GL_COUNT_UP_NV, 0x1F);

		glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	}

	_base_gl_vertex_array_empty.bind();

	_gl.program.output.use();
	glDrawArrays(GL_LINES, 0, (n_output_fragments + n_spans) * 2);
	_gl.program.output.disuse();

	if (_show_fps) {
		_base_gl_program_fps.use();
		glDrawArrays(GL_QUADS, 0, 4);
		_base_gl_program_fps.disuse();
	}

	_base_gl_vertex_array_empty.unbind();
	DEBUG_CHECK_GL_ERROR();

	if (_enable_step_timing) {
		glFinish();
		long long ts;
		QueryPerformanceCounter((LARGE_INTEGER*)&ts);
		_step_timestamp.push_back(ts);
	}

	//
	_dump_debug_data = false;
}

void VGRasterizer::initQMMaskTable() {

	std::vector<float2> sample_8;
	std::vector<float2> sample_32;

	float2 *gl_mask_8 = nullptr;
	float2 *gl_mask_32 = nullptr;

	if (_samples == 8) {
		gl_mask_8 = sample_8.data();
	}
	else {
		gl_mask_32 = sample_32.data();
	}

	QM_Mask_2x2::init_qm_mask_table(
		_samples,
		gl_mask_8, gl_mask_32,
		_sample_position,
		_qm_mask_table_pixel8, _qm_mask_table_pixel32);
}

void VGRasterizer::rasterizeImpl() {
	rasterizeImpl<VG_RASTERIZER_BIG_FRAGMENT_SIZE>();
}

} // end of namespace BigFragAM

} // end of namespace Mochimazui
