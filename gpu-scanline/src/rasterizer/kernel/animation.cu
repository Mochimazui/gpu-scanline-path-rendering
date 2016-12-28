
#include "animation.h"

#include <mochimazui/file.h>
#include <mochimazui/string.h>
#include <mochimazui/stdext.h>
#include <mochimazui/stdio_ext.h>

#include "../shared/ras_base.h"
#include "../shared/ras_define.h"
#include "../../cutil_math.h"
#include "../shared/ras_scan.h"
#include "../../bezier_curve_type.h"

namespace Mochimazui {

#include <mochimazui/3rd/stb_truetype.c>

__host__ __device__ int frgba_to_int(float r, float g, float b, float a) {
	return (((int)(255.f * a)) << 24)
		| (((int)(255.f * r)) << 16)
		| (((int)(255.f * g)) << 8)
		| (((int)(255.f * b)) << 0);
}

__host__ __device__ int hsv_color(float h, float c = 1.0f) {
	float x = 1.f - std::abs(std::fmod(h, 2.f) - 1.f);
	if ((0 <= h) && (h < 1)) { return frgba_to_int(c, x, 0.f, 1.f); }
	else if ((1 <= h) && (h < 2)) { return frgba_to_int(x, c, 0.f, 1.f); }
	else if ((2 <= h) && (h < 3)) { return frgba_to_int(0.f, c, x, 1.f); }
	else if ((3 <= h) && (h < 4)) { return frgba_to_int(0.f, x, c, 1.f); }
	else if ((4 <= h) && (h < 5)) { return frgba_to_int(x, 0.f, c, 1.f); }
	else if ((5 <= h) && (h < 6)) { return frgba_to_int(c, 0.f, x, 1.f); }
	else { return frgba_to_int(0.f, 0.f, 0.f, 1.f); }
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

namespace TreeAnimation {

#define NODE_NUMBER 0x2FFFF
//#define NODE_NUMBER 0x7FFF
//#define NODE_NUMBER 0x3FFF
//#define NODE_NUMBER 0xFFF
//#define NODE_NUMBER 0xF
//#define NODE_NUMBER 0x7
//#define NODE_NUMBER 0x3
//#define NODE_NUMBER 0x1

// -------- -------- -------- -------- -------- -------- -------- --------
__host__ __device__ int path_number_from_time(int time) {
	return NODE_NUMBER;
}

__host__ __device__ int curve_number_from_time(int time) {
	return (NODE_NUMBER)* 5;
}

__host__ __device__ int vertex_number_from_time(int time) {
	return curve_number_from_time(time) * 4;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_animation(
	int last_time,
	int next_time,

	float2 *p_vertex,
	uint32_t *p_vertex_path_id,

	uint32_t *p_curve_vertex_pos,
	uint8_t *p_curve_type,
	uint32_t *p_curve_path_id,
	uint32_t *p_curve_contour_id,
	float *p_curve_arc_w,
	float *p_curve_offset,
	uint8_t *p_curve_reversed,
	uint8_t *p_path_fill_rule,
	uint32_t *p_path_fill_info

	) {

	// one thread represents one segment.

	auto idx = GET_ID();
	if (idx >= NODE_NUMBER) { return; }

	//auto segment_id = idx + 1;

	//
	float2 seg_pos = make_float2(0.f, 0.f);
	float2 seg_dir = make_float2(0.f, 1.f);

	float len = 1.f;
	float width = 1.f;

#define ENABLE_GROW_UP
#ifdef ENABLE_GROW_UP

	const int prepare_time = 10000;
	const int linear_grow = 17000;
	const int sqrt_grow = 24000;

	if (next_time < prepare_time) {
		len = 0.05;
		width = 0.05;
	}
	else if (next_time < linear_grow) {
		len = 0.05 + 0.45 * (next_time - prepare_time) / (float)(linear_grow - prepare_time);
		width = 0.05 + 0.45 * (next_time - prepare_time) / (float)(linear_grow - prepare_time);
	}
	else if (next_time < sqrt_grow) {
		len = 0.5 + 0.5 * ((next_time - linear_grow) / (float)(sqrt_grow - linear_grow));
		width = 0.5 + 0.5 * ((next_time - linear_grow) / (float)(sqrt_grow - linear_grow));
	}
	else {
		width = 1.f + (next_time - 15000) / 1000.f * 0.01f;
	}

#endif

	float start_len = len;

	//
	float2 left_dir;
	float2 right_dir;
	//float2 parent_dir;

	float left_width;
	float right_width;
	//float parent_width;

	float left_len;
	float right_len;

	//
	//float2 v_l0, v_l1, v_r0, v_r1;
	//float2 v_l0, v_r0;
	float2 v_lc0, v_lc1, v_rc0, v_rc1;

	//float2 p_l1, p_r1;
	float2 p_lc1, p_rc1;

	//float2 cl_l0, cl_r0, cl_lc0, cl_rc0;
	//float2 cr_l0, cr_r0, cr_lc0, cr_rc0;
	float2 cl_lc0, cl_rc0;
	float2 cr_lc0, cr_rc0;

	// generate the mask
	uint32_t segment_id = idx + 1;
	uint32_t mask = 0xFFFFFFFFU;
	mask = segment_id == 1 ? 0 : mask >> (__clz(segment_id) + 1);

	auto animation_mask = 0xFFFFFFFF;
	auto raw_segment_id = segment_id;

	//bool is_root = segment_id == 1;

	auto seg_point = [](float2 pos, float2 tan, float width, int left_or_right) {
		auto norm = make_float2(-tan.y, tan.x);
		if (left_or_right == 1) {
			return pos + norm * width * 0.05f;
		}
		else {
			return pos - norm * width * 0.05f;
		}
	};

	auto seg_control_point = [](float2 pos, float2 tan, float width, float len, float t, int left_or_right) {
		auto norm = make_float2(-tan.y, tan.x);
		if (left_or_right == 1) {
			return pos + norm * width * 0.4f * 0.05f + tan * len * t;
		}
		else {
			return pos - norm * width * 0.4f  * 0.05f + tan * len * t;
		}
	};

	//p_l1 = p_lc1 = seg_point(seg_pos, seg_dir, width, 1);
	//p_r1 = p_rc1 = seg_point(seg_pos, seg_dir, width, 0);

	for (;;) {

		//

		auto aat = raw_segment_id & (~animation_mask);

		float aleft = 1.57f / 2.f + sin(next_time / (100.f + aat * 2) / 30.f) * 0.09;
		float aright = -1.57f / 2.f + sin(next_time / (90.f + aat * 3) / 30.f) * 0.11;

#ifdef ENABLE_GROW_UP
		if (next_time < prepare_time) {
			aleft = aright = 0;
		}
		else if (next_time < linear_grow) {
			aleft *= 0.5 * (next_time - prepare_time) / (float)(linear_grow - prepare_time);
			aright *= 0.5 * (next_time - prepare_time) / (float)(linear_grow - prepare_time);
		}
		else if (next_time < sqrt_grow) {
			aleft *= 0.5 + 0.5 * ((next_time - linear_grow) / (float)(sqrt_grow - linear_grow));
			aright *= 0.5 + 0.5 * ((next_time - linear_grow) / (float)(sqrt_grow - linear_grow));
		}
#endif

		//const float aleft = 1.57f / 2.f + sin(next_time / (aat * 2 / (float)(animation_mask))) * 0.10;
		//const float aright = -1.57f / 2.f + sin(next_time / (aat * 2 / (float)(animation_mask))) * 0.15;

		// this segment vertex
		//v_l0 = seg_point(seg_pos, seg_dir, width, 1);
		//v_l1 = v_l0 + seg_dir * len;
		//v_r0 = seg_point(seg_pos, seg_dir, width, 0);
		//v_r1 = v_r0 + seg_dir * len;

		v_lc0 = seg_control_point(seg_pos, seg_dir, width, len, 1 / 3.f, 1);
		v_lc1 = seg_control_point(seg_pos, seg_dir, width, len, 2 / 3.f, 1);
		v_rc0 = seg_control_point(seg_pos, seg_dir, width, len, 1 / 3.f, 0);
		v_rc1 = seg_control_point(seg_pos, seg_dir, width, len, 2 / 3.f, 0);

		// left
		float2 m0 = make_float2(cos(aleft), -sin(aleft));
		float2 m1 = make_float2(sin(aleft), cos(aleft));
		left_dir.x = dot(m0, seg_dir);
		left_dir.y = dot(m1, seg_dir);
		left_width = width * 0.81;
		left_len = (len == start_len ? len * 0.5f : len)* (0.78 + sin(next_time / (60.f + aat * 2) / 20.f) * 0.01);
		//left_len = (len == 1.f ? 0.5f : len)* 0.78;

		// right
		m0 = make_float2(cos(aright), -sin(aright));
		m1 = make_float2(sin(aright), cos(aright));
		right_dir.x = dot(m0, seg_dir);
		right_dir.y = dot(m1, seg_dir);
		right_width = width * 0.79;
		right_len = (len == start_len ? len * 0.5f : len) * (0.72 + sin(next_time / (50.f + aat * 3) / 20.f) * 0.01);
		//right_len = (len == 1.f ? 0.5f : len) * 0.72;

		// calculate chidren vertex.
		//cl_l0 = seg_point(seg_pos + seg_dir * len, left_dir, left_width, 1);
		//cl_r0 = seg_point(seg_pos + seg_dir * len, left_dir, left_width, 0);
		cl_lc0 = seg_control_point(seg_pos + seg_dir * len, left_dir, left_width, left_len, 1 / 3.f, 1);
		cl_rc0 = seg_control_point(seg_pos + seg_dir * len, left_dir, left_width, left_len, 1 / 3.f, 0);

		//cr_l0 = seg_point(seg_pos + seg_dir * len, right_dir, right_width, 1);
		//cr_r0 = seg_point(seg_pos + seg_dir * len, right_dir, right_width, 0);
		cr_lc0 = seg_control_point(seg_pos + seg_dir * len, right_dir, right_width, right_len, 1 / 3.f, 1);
		cr_rc0 = seg_control_point(seg_pos + seg_dir * len, right_dir, right_width, right_len, 1 / 3.f, 0);

		if (!mask) { break; }

		//

		if (segment_id & 1) {

			seg_pos = seg_pos + seg_dir * len;

			seg_dir = left_dir;
			len = left_len;
			width = left_width;

			//p_l1 = v_l1;
			p_lc1 = v_lc1;

			//p_r1 = cr_l0;
			p_rc1 = cr_lc0;
		}
		else {

			seg_pos = seg_pos + seg_dir * len;

			seg_dir = right_dir;
			len = right_len;
			width = right_width;

			//p_l1 = cl_r0;
			p_lc1 = cl_rc0;

			//p_r1 = v_r1;
			p_rc1 = v_rc1;
		}

		mask >>= 1;
		segment_id >>= 1;
		animation_mask <<= 1;
	}

	//
	auto path_id = idx;
	auto curve_id = idx * 5;
	auto curve_vertex_pos = idx * 14;

	// draw the quad.
	//auto start = seg_pos;

	//
	auto tan = seg_dir;
	auto norm = make_float2(-tan.y, tan.x);

	//start *= 300.f;
	//start.x += 500.f;
	//
	//len *= 300.f;
	//width *= 300.f;

	//auto v0 = lerp(v_l0, p_l1, .5f);
	//auto v1 = lerp(v_r0, p_r1, .5f);
	//auto v2 = lerp(v_r1, cr_r0, .5f);
	//auto v3 = lerp(v_l1, cl_l0, .5f);

	auto l0 = lerp(p_lc1, v_lc0, .5f);
	auto r0 = lerp(p_rc1, v_rc0, .5f);

	auto l1 = lerp(v_lc1, cl_lc0, .5f);
	auto r1 = lerp(v_rc1, cr_rc0, .5f);

	auto v_top = lerp(cl_rc0, cr_lc0, .5f);

	//auto cp_in = norm * (0.2f / 1000.f * 0.01f);
	//auto c0 = start + tan * len * 1.f / 3.f + cp_in * width * 0.05f;
	//auto c1 = start + tan * len * 1.f / 3.f - cp_in * width * 0.05f;
	//auto c2 = start + tan * len * 2.f / 3.f - cp_in * width * 0.05f;
	//auto c3 = start + tan * len * 2.f / 3.f + cp_in *width * 0.05f;

	auto scale = 200.f;
	auto translate = make_float2(500.f, 0.f);

	l0 = l0 * scale + translate;
	l1 = l1 * scale + translate;
	r0 = r0 * scale + translate;
	r1 = r1 * scale + translate;
	v_top = v_top * scale + translate;

	v_lc0 = v_lc0 * scale + translate;
	v_lc1 = v_lc1 * scale + translate;
	v_rc0 = v_rc0 * scale + translate;
	v_rc1 = v_rc1 * scale + translate;

#pragma unroll
	for (int i = 0; i < 5; ++i) {
		p_curve_reversed[curve_id + i] = 0;
		p_curve_path_id[curve_id + i] = path_id;
	}

#pragma unroll
	for (int i = 0; i < 14; ++i) {
		p_vertex_path_id[curve_vertex_pos + i] = path_id;
	}

	p_curve_type[curve_id + 0] = 2;
	p_curve_vertex_pos[curve_id + 0] = curve_vertex_pos + 0;
	p_vertex[curve_vertex_pos + 0] = l0;
	p_vertex[curve_vertex_pos + 1] = r0;

	p_curve_type[curve_id + 1] = 4;
	p_curve_vertex_pos[curve_id + 1] = curve_vertex_pos + 2;
	p_vertex[curve_vertex_pos + 2] = r0;
	p_vertex[curve_vertex_pos + 3] = v_rc0;
	p_vertex[curve_vertex_pos + 4] = v_rc1;
	p_vertex[curve_vertex_pos + 5] = r1;

	p_curve_type[curve_id + 2] = 2;
	p_curve_vertex_pos[curve_id + 2] = curve_vertex_pos + 6;
	p_vertex[curve_vertex_pos + 6] = r1;
	p_vertex[curve_vertex_pos + 7] = v_top;

	p_curve_type[curve_id + 3] = 2;
	p_curve_vertex_pos[curve_id + 3] = curve_vertex_pos + 8;
	p_vertex[curve_vertex_pos + 8] = v_top;
	p_vertex[curve_vertex_pos + 9] = l1;

	p_curve_type[curve_id + 4] = 4;
	p_curve_vertex_pos[curve_id + 4] = curve_vertex_pos + 10;
	p_vertex[curve_vertex_pos + 10] = l1;
	p_vertex[curve_vertex_pos + 11] = v_lc1;
	p_vertex[curve_vertex_pos + 12] = v_lc0;
	p_vertex[curve_vertex_pos + 13] = l0;

	//const int32_t color_table[16] = {
	//	0xFF003060,
	//	0xFF105080,
	//	0xFF2070A0,
	//	0xFF3080C0,
	//	0xFF4090D0,
	//	0xFF50A0E0,

	//	0xFF008500,
	//	0xFF009000,
	//	0xFF009500,
	//	0xFF00A000,
	//	0xF000A500,
	//	0xF000B000,
	//	0xF000B500,
	//	0xF000C000,
	//	0xF000C500,
	//};

	auto n_paths = path_number_from_time(next_time);
	if (idx < n_paths) {
		p_path_fill_rule[idx] = 0;
		p_path_fill_info[idx] =
			(31 - __clz(idx + 1)) <= 5 ? 0xFF3080C0 : 0xE000A500;
		//color_table[31 - __clz(idx + 1)];
		//hsv_color(6.0 * path_id / (float)NODE_NUMBER);
	}

}

} // end of namespace TreeAnimation


void tree_animation(
	int last_frame_timestamp,
	int next_frame_timestamp,
	RasterizerBase::VGInputCurveDataPack &_last_frame_curve_in,
	RasterizerBase::VGInputCurveDataPack &_next_frame_curve_in,
	RasterizerBase::VGInputPathDataPack &_last_frame_path_in,
	RasterizerBase::VGInputPathDataPack &_next_frame_path_in
	) {
	using namespace TreeAnimation;


	auto time = next_frame_timestamp;

	//
	auto n_paths = path_number_from_time(time);
	auto n_curves = curve_number_from_time(time);
	auto n_vertices = vertex_number_from_time(time);

	// check reserved space.
	if (n_paths >= _next_frame_path_in.fill_rule.reserved()) {
		printf("%d %d\n", n_paths, _next_frame_path_in.fill_rule.reserved());
		throw std::runtime_error("vg_animation: reserve more path");
	}

	if (n_curves >= _next_frame_curve_in.curve_type.reserved()) {
		printf("%d %d\n", n_curves, _next_frame_curve_in.curve_type.reserved());
		throw std::runtime_error("vg_animation: reserve more curve");
	}

	if (n_vertices >= _next_frame_curve_in.vertex.reserved()) {
		printf("%d %d\n", n_vertices, _next_frame_curve_in.vertex.reserved());
		throw std::runtime_error("vg_animation: reserve more vertex");
	}

	auto &_vg = _next_frame_curve_in;
	auto &_path = _next_frame_path_in;

	_vg.n_curves = n_curves;
	_vg.n_vertices = n_vertices;
	_path.n_paths = n_paths;

	//
	k_animation << < _vg.n_curves, 256 >> > (
		last_frame_timestamp,
		next_frame_timestamp,

		_vg.vertex,
		_vg.vertex_path_id,

		_vg.curve_vertex_pos,
		_vg.curve_type,
		_vg.curve_path_id,
		_vg.curve_contour_id,
		_vg.curve_arc_w,
		_vg.curve_offset,
		_vg.curve_reversed,

		_path.fill_rule,
		_path.fill_info
		);

	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("after animation");

}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

namespace ChordAnimation {

// -------- -------- -------- -------- -------- -------- -------- --------
__host__ __device__ int path_number_from_time(int time) {
	return time < 3000 ? 3 : (time + 999) / 1000;
}

__host__ __device__ int curve_number_from_time(int time) {
	return 3;
}

__host__ __device__ int vertex_number_from_time(int time) {
	return curve_number_from_time(time) * 4;
}

struct ChordAnimationContext {

	CUDATL::CUDAArray<int32_t> node_color;
	CUDATL::CUDAArray<float> node_size;
	CUDATL::CUDAArray<float> node_pos;

	CUDATL::CUDAArray<float> node_t_0;
	CUDATL::CUDAArray<float> node_t_1;

	CUDATL::CUDAArray<int> node_link_number;
	CUDATL::CUDAArray<int> node_link_table;

	// every 4 : (node_0, node_1, num_0, num_1
	CUDATL::CUDAArray<uint32_t> link_node_0;
	CUDATL::CUDAArray<uint32_t> link_node_1;

	CUDATL::CUDAArray<float> link_size_0;
	CUDATL::CUDAArray<float> link_size_1;

	CUDATL::CUDAArray<float> link_pos_0;
	CUDATL::CUDAArray<float> link_pos_1;

	CUDATL::CUDAArray<float> link_t_0;
	CUDATL::CUDAArray<float> link_t_1;

	void reserve(int n_node, int n_link) {

		n_node += 1;

		node_color.resizeWithoutCopy(n_node);
		node_size.resizeWithoutCopy(n_node);
		node_pos.resizeWithoutCopy(n_node);
		//node_t.resizeWithoutCopy(n_node * 2);

		link_node_0.resizeWithoutCopy(n_link);
		link_node_1.resizeWithoutCopy(n_link);
		link_size_0.resizeWithoutCopy(n_link);
		link_size_1.resizeWithoutCopy(n_link);
		link_pos_0.resizeWithoutCopy(n_link);
		link_pos_1.resizeWithoutCopy(n_link);
		//link_t.resizeWithoutCopy(n_link * 4);
	}

};

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_gen_current_chord(
	int last_link_number,
	int current_link_number,

	int time,

	float *i_link_size_0,
	float *i_link_size_1,

	float *o_link_size_0,
	float *o_link_size_1
	) {

	auto idx = GET_ID();

	if (idx >= current_link_number) { return; }
	if (idx < last_link_number) {
		o_link_size_0[idx] = i_link_size_0[idx];
		o_link_size_1[idx] = i_link_size_1[idx];
	}
	else {
		auto t = (( (time-1) % 1000) + 1) / 1000.f;
		o_link_size_0[idx] = i_link_size_0[idx] * t;
		o_link_size_1[idx] = i_link_size_1[idx] * t;
	}

}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_add_new_link_to_table(int link_start, int link_number,

	uint32_t *link_node_0,
	uint32_t *link_node_1,

	int32_t *node_link_number,
	int32_t *node_link_table
	)
{
	auto idx = GET_ID();
	if (idx >= link_number) { return; }

	auto link_id = link_start + idx;

	auto node_0 = link_node_0[link_id];
	auto node_1 = link_node_1[link_id];

	auto index_0 = atomicAdd(node_link_number + node_0, 1);
	auto index_1 = atomicAdd(node_link_number + node_1, 1);

	node_link_table[node_0 * 4096 + index_0] = link_id;
	node_link_table[node_1 * 4096 + index_1] = link_id;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_count_node_size(

	int n_nodes,

	int32_t *node_link_number,
	int32_t *node_link_table,

	float *i_link_size_0,
	float *i_link_size_1,

	uint32_t *i_link_node_0,
	uint32_t *i_link_node_1,

	float *o_node_size,
	float *o_link_t
	) {

	auto idx = GET_ID();

	if (idx >= n_nodes) { return; }

	float size = 0;

	float link_number = node_link_number[idx];

	for (int  i = 0; i < link_number;++i) {

		auto link_id = node_link_table[idx * 4096 + i];

		auto link_node_0 = i_link_node_0[link_id];
		//auto link_node_1 = i_link_node_1[link_id];

		auto link_size_0 = i_link_size_0[link_id];
		auto link_size_1 = i_link_size_1[link_id];

		if (link_node_0 == idx) {
			o_link_t[link_id * 2] = size;
			size += link_size_0;
		}
		else {
			o_link_t[link_id * 2 + 1] = size;
			size += link_size_1;
		}
	}
	o_node_size[idx] = size;
}

//// -------- -------- -------- -------- -------- -------- -------- --------
//__global__ void k_count_node_size(
//	int n_links,
//
//	float *i_link_size_0,
//	float *i_link_size_1,
//
//	uint32_t *i_link_node_0,
//	uint32_t *i_link_node_1,
//
//	float *o_node_size,
//	float *o_link_t
//	) {
//
//	auto idx = GET_ID();
//
//	if (idx >= n_links) { return; }
//
//	auto id0 = i_link_node_0[idx];
//	auto id1 = i_link_node_1[idx];
//
//	auto size0 = i_link_size_0[idx];
//	auto size1 = i_link_size_1[idx];
//
//	auto pos_0 = atomicAdd(o_node_size + id0, size0);
//	auto pos_1 = atomicAdd(o_node_size + id1, size1);
//
//	o_link_t[idx * 2] = pos_0;
//	o_link_t[idx * 2 + 1] = pos_1;
//}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

struct Link {
	int id_0 = -1;
	int id_1 = -1;

	float size_0 = 0.f;
	float size_1 = 0.f;
	float pos_0 = 0.f;
	float pos_1 = 0.f;

	float pf_size_0 = 0.f;
	float pf_size_1 = 0.f;
	float pf_pos_0 = 0.f;
	float pf_pos_1 = 0.f;

	int64_t key() {
		if (id_0 <= id_1) {
			return (((int64_t)id_0) << 32) | id_1;
		}
		else {
			return (((int64_t)id_1) << 32) | id_0;
		}
	}

};

struct Graph {

public:
	void resize(int n_nodes) {
		_n_nodes = n_nodes;
		_link.resize(_n_nodes * n_nodes);
	}

	Link &link(int n0, int n1) {
		if (n0 <= n1) { return _link[n0*_n_nodes + n1]; }
		else { return link(n1, n0); }
	}

	const Link &link(int n0, int n1) const {
		if (n0 <= n1) { return _link[n0*_n_nodes + n1]; }
		else { return link(n1, n0); }
	}

private:

	int _n_nodes = 0;
	std::vector<Link> _link;
	
};

struct Node {
	int id = 0;
	float size = 0.f;
	float pos = 0.f;
	//
	float t0 = 0.f;
	float t1 = 0.f;
	//
	bool operator < (const Node &o) {
		return size < o.size;
	}
};

// -------- -------- -------- -------- -------- -------- -------- --------
static const int color_table[36] = {
	0xFF9C6744,
	0xFFC9BEB9,
	0xFFCFA07E,
	0xFFC4BAA1,
	0xFFC2B6BF,
	0xFF8FB5AA,
	0xFF85889E,
	0xFF9C7989,
	0xFF91919C,
	0xFF99677B,
	0xFF918A59,
	0xFF6E676C,
	0xFF6E4752,
	0xFF6B4A2F,
	0xFF998476,
	0xFF8A968D,
	0xFF968D8A,
	0xFF968D96,
	0xFFCC855C,
	0xFF967860,
	0xFF929488,
	0xFF949278,
	0xFFA0A3BD,
	0xFFBD93A1,
	0xFF65666B,
	0xFF6B5745,
	0xFF6B6664,
	0xFF695C52,
	0xFF56695E,
	0xFF69545C,
	0xFF565A69,
	0xFF696043,
	0xFF63635C,
	0xFF636150,
	0xFFCFB6A3,
	0xFF666655,
};

// -------- -------- -------- -------- -------- -------- -------- --------
struct RefugeeFrameDrawData {

	// node data.
	CUDATL::CUDAArray<float> node_t_0;
	CUDATL::CUDAArray<float> node_t_1;
	CUDATL::CUDAArray<float> node_size;
	CUDATL::CUDAArray<int32_t> node_color;

	// link data

	CUDATL::CUDAArray<uint32_t> link_node_0;
	CUDATL::CUDAArray<uint32_t> link_node_1;

	// previous frame data.
	CUDATL::CUDAArray<float> link_pf_size_0;
	CUDATL::CUDAArray<float> link_pf_size_1;

	CUDATL::CUDAArray<float> link_pf_pos_0;
	CUDATL::CUDAArray<float> link_pf_pos_1;

	// this frame data.
	CUDATL::CUDAArray<float> link_size_0;
	CUDATL::CUDAArray<float> link_size_1;

	CUDATL::CUDAArray<float> link_pos_0;
	CUDATL::CUDAArray<float> link_pos_1;

	void reserve(int n_nodes, int n_links) {
		node_t_0.resizeWithoutCopy(n_nodes);
		node_t_1.resizeWithoutCopy(n_nodes);
		node_size.resizeWithoutCopy(n_nodes);

		link_size_0.resizeWithoutCopy(n_links);
		link_size_1.resizeWithoutCopy(n_links);
		link_pos_0.resizeWithoutCopy(n_links);
		link_pos_1.resizeWithoutCopy(n_links);
	}
};

struct RefugeeFrameData {

	// cpu data
	std::vector<Node> nodes;
	Graph graph;

	RefugeeFrameDrawData t0;
	RefugeeFrameDrawData t1;

};

// -------- -------- -------- -------- -------- -------- -------- --------

void* _hfont = 0;

struct TextContext {

	//load the font
	void init(int reserve_curve)  {

		if (!_hfont) {
			//FILE* hf = fopen("font/cmunrm.ttf", "rb");
			FILE* hf = fopen("font/Righteous-Regular.ttf", "rb");

			if (!hf) {
				_hfont = NULL;
			}
			else {
				fseek(hf, 0, SEEK_END);
				int sz = ftell(hf);
				char* font_data = new char[sz];
				fseek(hf, 0, SEEK_SET);
				fread(font_data, 1, sz, hf);
				_hfont = (void*)FTW_LoadFont(font_data, sz);
				fclose(hf);
				//we need to keep the pointer... currently... let's go for the leak
			}
		}

		// allocate memory
		//int reserve_curve = 50000;
		_fill_in.curve_vertex_pos.resizeWithoutCopy(reserve_curve);
		_fill_in.curve_type.resizeWithoutCopy(reserve_curve);
		_fill_in.curve_path_id.resizeWithoutCopy(reserve_curve);
		_fill_in.vertex.resizeWithoutCopy(reserve_curve * 4);
		_fill_in.vertex_path_id.resizeWithoutCopy(reserve_curve * 4);
	}

	void startTextInput(int x, int y, float h_font, uint32_t color) {
		if (!_hfont) { return; }

		//if (!_text_input_contexts.empty()) {
		//	insertChar('\b');
		//}

		//_text_input_contexts.clear();
		// insert a '\n'
		RasterizerBase::TTextInputContext ctx;
		ctx.x = (float)x;
		ctx.y = (float)y;
		ctx.h_font = h_font;
		ctx.color = color;
		ctx.ch_last = '\n';

		ctx.curve_index = _nextCurveIndex;
		//ctx.path_index = _nextPathIndex;
		ctx.vertex_index = _nextVertexIndex;

		_text_input_contexts.clear();
		_text_input_contexts.push_back(ctx);
	}

	void stopTextInput() {
		insertChar('\b'); 
	}

	void insertChar(int ch, int path_id = 0) {

		// -------- -------- -------- --------
		// bind old input pack
		auto &_in = _fill_in;

		if (ch == '\b') {
			//backspace
			if (_text_input_contexts.size() <= 1) { return; }

			auto ctx = _text_input_contexts.back();

			_path_in.n_paths = ctx.path_index;
			_in.n_curves = ctx.curve_index;
			_in.n_vertices = ctx.vertex_index;

			//_nextPathIndex = _path_in.n_paths;
			_nextCurveIndex = _in.n_curves;
			_nextVertexIndex = _in.n_vertices;

			_text_input_contexts.pop_back();
		}
		else {
			RasterizerBase::TTextInputContext ctx = _text_input_contexts.back();
			//ligatures
			if (ctx.ch_last == 'f'&&ch == 'f') {
				insertChar('\b');
				insertChar(0xfb00);
				return;
			}
			if (ctx.ch_last == 'f'&&ch == 'i') {
				insertChar('\b');
				insertChar(0xfb01);
				return;
			}
			if (ctx.ch_last == 'f'&&ch == 'l') {
				insertChar('\b');
				insertChar(0xfb02);
				return;
			}
			if (ctx.ch_last == 0xfb00 && ch == 'i') {
				insertChar('\b');
				insertChar(0xfb03);
				return;
			}
			if (ctx.ch_last == 0xfb00 && ch == 'l') {
				insertChar('\b');
				insertChar(0xfb04);
				return;
			}
			//insert a new char
			//RasterizerBase::TTextInputContext ctx0 = ctx;
			float scale = stbtt_ScaleForPixelHeight((FT_Face)_hfont, ctx.h_font);
			if (ch != ' ') {
				stbtt_vertex* vertices = NULL;
				int nv = stbtt_GetCodepointShape((FT_Face)_hfont, ch, &vertices);

				if (vertices) {
					
					std::vector<int> curve_vertex_pos;
					std::vector<uint8_t> curve_type;
					std::vector<int> curve_path_id;
					std::vector<int> vertex_path_id;
					std::vector<float2> vertex_point;

					//x,y,cx,cy,type
					float ins_x = 0.f, ins_y = 0.f;
					for (int i = 0; i<nv; i++) {
						float vx = (float)vertices[i].x*scale + ctx.x;
						float vy = (float)vertices[i].y*scale + ctx.y;
						switch (vertices[i].type) {
						default:
							assert(0);
							break;
						case STBTT_vmove:
							ins_x = vx;
							ins_y = vy;
							break;
						case STBTT_vline:
							//we must add two interleaved curves at a time
							curve_vertex_pos.push_back(_nextVertexIndex + (int)vertex_point.size());
							curve_path_id.push_back(path_id);
							curve_type.push_back(CT_Linear);
							vertex_point.push_back(make_float2(vx, vy));
							vertex_point.push_back(make_float2(ins_x, ins_y));
							vertex_path_id.push_back(path_id);
							vertex_path_id.push_back(path_id);
							ins_x = vx;
							ins_y = vy;
							break;
						case STBTT_vcurve:
							curve_vertex_pos.push_back(_nextVertexIndex + (int)vertex_point.size());
							curve_path_id.push_back(path_id);
							curve_type.push_back(CT_Quadratic);
							vertex_point.push_back(make_float2(vx, vy));
							vertex_point.push_back(make_float2((float)vertices[i].cx*scale + ctx.x, (float)vertices[i].cy*scale + ctx.y));
							vertex_point.push_back(make_float2(ins_x, ins_y));
							vertex_path_id.push_back(path_id);
							vertex_path_id.push_back(path_id);
							vertex_path_id.push_back(path_id);
							ins_x = vx;
							ins_y = vy;
							break;
						}
					}
					STBTT_free(vertices, 0);

					cudaMemcpy(_in.curve_vertex_pos.gptr() + _nextCurveIndex,
						curve_vertex_pos.data(), 4 * curve_vertex_pos.size(),
						cudaMemcpyHostToDevice);

					cudaMemcpy(_in.curve_type.gptr() + _nextCurveIndex,
						curve_type.data(), curve_type.size(),
						cudaMemcpyHostToDevice);

					cudaMemcpy(_in.curve_path_id.gptr() + _nextCurveIndex,
						curve_path_id.data(), 4 * curve_path_id.size(),
						cudaMemcpyHostToDevice);

					cudaMemcpy(_in.vertex.gptr() + _nextVertexIndex,
						vertex_point.data(), sizeof(float2) * vertex_point.size(),
						cudaMemcpyHostToDevice);

					cudaMemcpy(_in.vertex_path_id.gptr() + _nextVertexIndex,
						vertex_path_id.data(), 4 * vertex_path_id.size(),
						cudaMemcpyHostToDevice);

					//ctx.path_index = _nextPathIndex;
					ctx.curve_index = _nextCurveIndex;
					ctx.vertex_index = _nextVertexIndex;

					//++_nextPathIndex;
					_nextCurveIndex += (int)curve_type.size();
					_nextVertexIndex += (int)vertex_point.size();

					//_path_in.n_paths = _nextPathIndex;
					_in.n_curves = _nextCurveIndex;
					_in.n_vertices = _nextVertexIndex;
				}
			}

			//advance the cursor
			ctx.x += FTW_GetCharAdvance((FT_Face)_hfont, ctx.h_font, ch) +
				FTW_GetKerning((FT_Face)_hfont, ctx.h_font, ctx.ch_last, ch);
			ctx.ch_last = ch;
			_text_input_contexts.push_back(ctx);
		}
	}

public:

	std::vector<RasterizerBase::TTextInputContext> _text_input_contexts;

	RasterizerBase::VGInputCurveDataPack _fill_in;
	RasterizerBase::VGInputPathDataPack _path_in;

	//int _nextPathIndex = 0;
	int _nextCurveIndex = 0;
	int _nextVertexIndex = 0;
};

std::shared_ptr<TextContext> g_text_ctx;
std::shared_ptr<TextContext> g_year_text_ctx;
int number_of_years = 0;
std::vector<int> g_year_colors;

// -------- -------- -------- -------- -------- -------- -------- --------
class RefugeeAnimation {
public:
	void setNodeNumber(int n) { _n_nodes = n; }
	void reserve(int n) { _frames.reserve(n); }
public:
	void addFrame(std::map<int64_t, Link> &links);
	void genAnimation();
	RefugeeFrameData &frame(int i) { return _frames[i]; }
private:
	void calculatePosition (int frame_id);
	void genFrame(int frame_id);
public:
	CUDATL::CUDAArray<float> node_name_width;
private:
	int _n_nodes;
	std::vector<RefugeeFrameData> _frames;
	
};

void RefugeeAnimation::addFrame(std::map<int64_t, Link> &in_links) {

	_frames.push_back(RefugeeFrameData());
	auto &frame = _frames.back();
	auto &graph = frame.graph;

	graph.resize(_n_nodes);

	for (auto &ilink : in_links) {
		auto &link = ilink.second;
		auto &gl = graph.link(link.id_0, link.id_1);

		if (link.id_0 == link.id_1) {
			auto sum = link.size_0 + link.size_1;
			link.size_0 = sum;
			link.size_1 = 0.f;
		}

		gl.id_0 = link.id_0;
		gl.id_1 = link.id_1;
		gl.size_0 = link.size_0;
		gl.size_1 = link.size_1;

		//if (link.id_0 <= link.id_1) {
		//	gl.id_0 = link.id_0;
		//	gl.id_1 = link.id_1;
		//	gl.size_0 = link.size_0;
		//	gl.size_1 = link.size_1;
		//}
		//else {
		//	gl.id_0 = link.id_0;
		//	gl.id_1 = link.id_1;
		//	gl.size_0 = link.size_1;
		//	gl.size_1 = link.size_0;
		//}
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void RefugeeAnimation::genAnimation() {

	// update frame link pos & node pos.
	for (int i = 0; i < _frames.size(); ++i) {
		calculatePosition(i);
	}

	// gen frame
	for (int i = 1; i < _frames.size(); ++i) {
		genFrame(i);
	}

};

// -------- -------- -------- -------- -------- -------- -------- --------
void RefugeeAnimation::calculatePosition(int frame_id) {

	auto &frame = _frames[frame_id];
	auto &graph = frame.graph;
	auto &nodes = frame.nodes;
	auto n_nodes = _n_nodes;
	if (nodes.size() != n_nodes) { nodes.resize(n_nodes); }

	// udpate position and size
	for (int i = 0; i < nodes.size(); ++i) { nodes[i].id = i; }

	for (int i = 0; i < n_nodes; ++i) {
		for (int j = 0; j <=i ; ++j) {
			auto &link = graph.link(j, i);

			if (link.id_0 == -1 || link.id_1 == -1) { continue; }

			if (link.id_0 == link.id_1) {
				link.pos_0 = nodes[link.id_0].size;
				link.pos_1 = nodes[link.id_1].size;

				nodes[link.id_0].size += link.size_0;
				//nodes[link.id_1].size = link.size_0 / 2.f;
			}
			else {
				link.pos_0 = nodes[link.id_0].size;
				link.pos_1 = nodes[link.id_1].size;

				nodes[link.id_0].size += link.size_0;
				nodes[link.id_1].size += link.size_1;
			}

		}
	}

	std::sort(nodes.begin(), nodes.end(), [](const Node &a, const Node &b) {
		return a.size > b.size;
	});

	float total_size = 0.f;
	for (int i = 0; i < nodes.size(); ++i) {
		total_size += nodes[i].size;
		if (i) {
			nodes[i].pos = nodes[i - 1].pos + nodes[i - 1].size;
		}
	}

	int no_empty_node = 0;
	for (int i = 0; i < nodes.size(); ++i) {
		auto size = nodes[i].size;
		if (size) { ++no_empty_node; }
	}
	//
	auto pi = acos(-1.f);
	auto gap = pi / 180.f * 2.0f;
	auto total_gap = gap * no_empty_node;

	int before = 1;
	for (int i = 0; i < nodes.size(); ++i) {
		auto pos = nodes[i].pos;
		auto size = nodes[i].size;
		if (size != 0.f) {
			++before;
		}
		nodes[i].t0 = before * gap + pos / total_size * (2 * pi - total_gap);
		nodes[i].t1 = before * gap + (pos + size) / total_size * (2 * pi - total_gap);
	}

	std::sort(nodes.begin(), nodes.end(), [](const Node &a, const Node &b) {
		return a.id < b.id;
	});

}

// -------- -------- -------- -------- -------- -------- -------- --------
void RefugeeAnimation::genFrame(int frame_id) {

	auto n_nodes = _n_nodes;

	// -------- -------- -------- --------
	// update color table
	static std::vector<int32_t> cpu_node_color;
	while (cpu_node_color.size() < n_nodes) {
		auto color = color_table[rand() % 36];
		auto t0 = color & 0x00FF0000;
		auto t1 = color & 0x000000FF;
		color = color & 0xFF00FF00;
		color |= t0 >> 16;
		color |= t1 << 16;
		cpu_node_color.push_back(color);
	}

	// -------- -------- -------- --------
	// -------- -------- -------- --------

	// nodes

	// previous frame nodes.
	auto &frame = _frames[frame_id];

	{
		std::vector<float> cpu_node_size;
		std::vector<float> cpu_node_t_0;
		std::vector<float> cpu_node_t_1;

		auto &prev_frame = _frames[frame_id - 1];

		for (auto &node : prev_frame.nodes) {
			cpu_node_size.push_back(node.size);
			cpu_node_t_0.push_back(node.t0);
			cpu_node_t_1.push_back(node.t1);
		}

		frame.t0.node_size.set(cpu_node_size);
		frame.t0.node_t_0.set(cpu_node_t_0);
		frame.t0.node_t_1.set(cpu_node_t_1);
		frame.t0.node_color.set(cpu_node_color);

	}

	// current frame nodes.
	{
		std::vector<float> cpu_node_size;
		std::vector<float> cpu_node_t_0;
		std::vector<float> cpu_node_t_1;

		for (auto &node : frame.nodes) {

			cpu_node_size.push_back(node.size);
			cpu_node_t_0.push_back(node.t0);
			cpu_node_t_1.push_back(node.t1);
		}

		frame.t1.node_size.set(cpu_node_size);
		frame.t1.node_t_0.set(cpu_node_t_0);
		frame.t1.node_t_1.set(cpu_node_t_1);
		frame.t1.node_color.set(cpu_node_color);
	}

	// -------- -------- -------- --------
	// -------- -------- -------- --------

	// links

	// prev frame
	{
		auto &prev_frame = _frames[frame_id - 1];

		std::vector<uint32_t> cpu_link_node_0;
		std::vector<uint32_t> cpu_link_node_1;

		std::vector<float> cpu_link_size_0;
		std::vector<float> cpu_link_size_1;

		std::vector<float> cpu_link_pos_0;
		std::vector<float> cpu_link_pos_1;

		for (int i = 0; i < n_nodes; ++i) {
			for (int j = 0; j <= i; ++j) {
				auto &prev_link = prev_frame.graph.link(j, i);
				auto &link = frame.graph.link(j, i);

				if (prev_link.size_0 != 0 || prev_link.size_1 != 0
					|| link.size_0 != 0 || link.size_1 != 0) {

					// add
					if (prev_link.id_0 == -1) {
						cpu_link_node_0.push_back(link.id_0);
						cpu_link_node_1.push_back(link.id_1);

						cpu_link_size_0.push_back(0.f);
						cpu_link_size_1.push_back(0.f);

						cpu_link_pos_0.push_back(0.f);
						cpu_link_pos_1.push_back(0.f);
					}
					else {
						cpu_link_node_0.push_back(prev_link.id_0);
						cpu_link_node_1.push_back(prev_link.id_1);

						cpu_link_size_0.push_back(prev_link.size_0);
						cpu_link_size_1.push_back(prev_link.size_1);

						cpu_link_pos_0.push_back(prev_link.pos_0);
						cpu_link_pos_1.push_back(prev_link.pos_1);
					}
					
				}
			}
		}

		frame.t0.link_node_0.set(cpu_link_node_0);
		frame.t0.link_node_1.set(cpu_link_node_1);

		frame.t0.link_size_0.set(cpu_link_size_0);
		frame.t0.link_size_1.set(cpu_link_size_1);

		frame.t0.link_pos_0.set(cpu_link_pos_0);
		frame.t0.link_pos_1.set(cpu_link_pos_1);
	}

	// prev frame
	{
		auto &prev_frame = _frames[frame_id - 1];

		std::vector<uint32_t> cpu_link_node_0;
		std::vector<uint32_t> cpu_link_node_1;

		std::vector<float> cpu_link_size_0;
		std::vector<float> cpu_link_size_1;

		std::vector<float> cpu_link_pos_0;
		std::vector<float> cpu_link_pos_1;

		for (int i = 0; i < n_nodes; ++i) {
			for (int j = 0; j <= i; ++j) {
				auto &prev_link = prev_frame.graph.link(j, i);
				auto &link = frame.graph.link(j, i);

				if (prev_link.size_0 != 0 || prev_link.size_1 != 0
					|| link.size_0 != 0 || link.size_1 != 0) {

					// add
					if (link.id_0 == -1) {
						cpu_link_node_0.push_back(prev_link.id_0);
						cpu_link_node_1.push_back(prev_link.id_1);

						cpu_link_size_0.push_back(0.f);
						cpu_link_size_1.push_back(0.f);

						// todo: find previous link pos
						cpu_link_pos_0.push_back(0.f);
						cpu_link_pos_1.push_back(0.f);
					}
					else {
						cpu_link_node_0.push_back(link.id_0);
						cpu_link_node_1.push_back(link.id_1);

						cpu_link_size_0.push_back(link.size_0);
						cpu_link_size_1.push_back(link.size_1);

						cpu_link_pos_0.push_back(link.pos_0);
						cpu_link_pos_1.push_back(link.pos_1);
					}

				}
			}
		}

		frame.t1.link_node_0.set(cpu_link_node_0);
		frame.t1.link_node_1.set(cpu_link_node_1);

		frame.t1.link_size_0.set(cpu_link_size_0);
		frame.t1.link_size_1.set(cpu_link_size_1);

		frame.t1.link_pos_0.set(cpu_link_pos_0);
		frame.t1.link_pos_1.set(cpu_link_pos_1);
	}

#ifdef NEW_ANIMATION

	frame.cpu_links = in_links;

	
	// -------- -------- -------- --------
	// update link previous frame data.
	if (_frames.size() > 1) {
		auto &pf_links = _frames[_frames.size() - 2];
		for (auto &link : links) {
			auto pfl = pf_links.cpu_links.find(link.first);
			if (pfl != pf_links.cpu_links.end()) {
				link.second.pf_pos_0 = pfl->second.pos_0;
				link.second.pf_pos_1 = pfl->second.pos_1;
				link.second.pf_size_0 = pfl->second.size_0;
				link.second.pf_size_1 = pfl->second.size_1;
			}
		}
	}

	// -------- -------- -------- --------
	
	// -------- -------- -------- --------
	// generate HOST data array
	std::vector<uint32_t> cpu_link_node_0;
	std::vector<uint32_t> cpu_link_node_1;
	
	std::vector<float> cpu_link_size_0;
	std::vector<float> cpu_link_size_1;

	std::vector<float> cpu_link_pos_0;
	std::vector<float> cpu_link_pos_1;

	std::vector<float> cpu_pf_link_size_0;
	std::vector<float> cpu_pf_link_size_1;

	std::vector<float> cpu_pf_link_pos_0;
	std::vector<float> cpu_pf_link_pos_1;

	
	for (auto &ilink : links) {
		auto &link = ilink.second;
		cpu_link_node_0.push_back(link.id_0);
		cpu_link_node_1.push_back(link.id_1);

		cpu_link_size_0.push_back(link.size_0);
		cpu_link_size_1.push_back(link.size_1);

		cpu_link_pos_0.push_back(link.pos_0);
		cpu_link_pos_1.push_back(link.pos_1);

		cpu_pf_link_size_0.push_back(link.pf_size_0);
		cpu_pf_link_size_1.push_back(link.pf_size_1);

		cpu_pf_link_pos_0.push_back(link.pf_pos_0);
		cpu_pf_link_pos_1.push_back(link.pf_pos_1);
	}

	
	// -------- -------- -------- --------
	// copy data to GMEMini
	frame.link_node_0.set(cpu_link_node_0);
	frame.link_node_1.set(cpu_link_node_1);

	frame.link_size_0.set(cpu_link_size_0);
	frame.link_size_1.set(cpu_link_size_1);

	frame.link_pos_0.set(cpu_link_pos_0);
	frame.link_pos_1.set(cpu_link_pos_1);

	frame.link_pf_size_0.set(cpu_pf_link_size_0);
	frame.link_pf_size_1.set(cpu_pf_link_size_1);

	frame.link_pf_pos_0.set(cpu_pf_link_pos_0);
	frame.link_pf_pos_1.set(cpu_pf_link_pos_1);

#endif
}

// -------- -------- -------- -------- -------- -------- -------- --------
void load_refugee_animation(
	std::shared_ptr<RefugeeAnimation> &source_country,
	std::shared_ptr<RefugeeAnimation> &target_country
	) {

	//
	Mochimazui::stdext::string data;
	Mochimazui::readAll("./chord_data/refugee.csv", data);
	//Mochimazui::readAll("./chord_data/refugee-test.csv", data);
	//Mochimazui::readAll("./chord_data/refugee-1975.csv", data);
	//Mochimazui::readAll("./chord_data/refugee-1975-1978.csv", data);

	printf("Gen animation\n");

	//
	auto lines = data.splitLine();

	struct Record {
		int year;
		//
		std::string name_0;
		std::string name_1;
		int id_0;
		int id_1;
		//
		float value_0;
		float value_1;
		//
		bool operator <(const Record &o) {
			return year < o.year;
		}
	};

	std::vector<Record> records;

	for (auto &line : lines) {
		auto values = line.split(',');

		Record record;

		record.year = values[0].toInt32();

		record.name_0 = values[1];
		record.name_1 = values[2];

		record.value_0 = values[3].toFloat();
		record.value_1 = values[4].toFloat();

		if (record.value_0 < 1000 && record.value_1 < 1000) { continue; }

		if (record.name_0 == record.name_1) {
			auto sum = record.value_0 + record.value_1;
			record.value_0 = sum;
			record.value_1 = 0.f;
		}
		else {
			if (record.value_0 == 0.f) { record.value_0 = 1.f; }
			if (record.value_1 == 0.f) { record.value_1 = 1.f; }
		}
		
		records.push_back(record);
	}

	std::sort(records.begin(), records.end());

	//
	std::vector<std::vector<Record>> records_by_year;
	std::map<std::string, int> name_to_id;

	g_text_ctx.reset(new TextContext);
	g_year_text_ctx.reset(new TextContext);

	g_text_ctx->init(50000);
	g_year_text_ctx->init(10000);

	//
	auto last_year = -1;
	for (auto &record : records) {

		// gemerate id
		auto i0 = name_to_id.find(record.name_0);
		record.id_0 = i0 != name_to_id.end() ? i0->second : name_to_id[record.name_0] = (int)name_to_id.size();
		auto i1 = name_to_id.find(record.name_1);
		record.id_1 = i1 != name_to_id.end() ? i1->second : name_to_id[record.name_1] = (int)name_to_id.size();

		//
		if (record.year != last_year) {
			last_year = record.year;

			auto pid = (int)records_by_year.size();
			g_year_text_ctx->startTextInput(0, 0, 128, 0xFFFFFFFF);
			g_year_text_ctx->insertChar( (last_year / 1000) + '0', pid);
			g_year_text_ctx->insertChar( ((last_year / 100) % 10)+ '0', pid);
			g_year_text_ctx->insertChar( ((last_year / 10) % 10) + '0', pid);
			g_year_text_ctx->insertChar( (last_year % 10) + '0', pid);

			records_by_year.push_back(std::vector<Record>());

		}
		records_by_year.back().push_back(record);
	}

	{
		auto year_n_curve = g_year_text_ctx->_nextCurveIndex;
		auto year_n_vertex = g_year_text_ctx->_nextVertexIndex;

		if (year_n_curve >= g_year_text_ctx->_fill_in.curve_vertex_pos.size()) {
			printf("!!! year text require more curve %d !!!\n", year_n_curve);
		}

		if (year_n_vertex >= g_year_text_ctx->_fill_in.vertex.size()) {
			printf("!!! year text require more vertex %d !!!\n", year_n_vertex);
		}
	}

	number_of_years = (int)records_by_year.size();
	while (g_year_colors.size() < number_of_years) {
		g_year_colors.push_back(0xFF7F7F7F);
	}

	auto find_or_create = [](std::map<int64_t, Link> &lmap, int64_t key) -> Link& {
		if (lmap.find(key) == lmap.end()) {
			Link new_link;
			new_link.id_0 = (int32_t)(key >> 32);
			new_link.id_1 = (int32_t)(key & 0xFFFFFFFF);
			return lmap[key] = new_link;
		}
		return lmap[key];
	};

	
	std::vector<std::pair<int, std::string>> _texts;

	int n_paths = (int)name_to_id.size();

	for (auto &i : name_to_id) {
		_texts.push_back(make_pair(i.second, i.first));
	}

	std::sort(_texts.begin(), _texts.end(), [](
		const std::pair<int, std::string>& a,
		const std::pair<int, std::string>& b
		) {
		return a.first < b.first;
	});

	
	std::vector<float> country_name_width;
	for (auto &i : _texts) {
		g_text_ctx->startTextInput(0, 0, 24.f, 0xFFFFFFFF);
		for (auto c : i.second) {
			g_text_ctx->insertChar(c, i.first);
		}
		country_name_width.push_back(g_text_ctx->_text_input_contexts.back().x);
	}

	{
		auto text_n_curve = g_text_ctx->_nextCurveIndex;
		auto text_n_vertex = g_text_ctx->_nextVertexIndex;

		if (text_n_curve >= g_text_ctx->_fill_in.curve_vertex_pos.size()) {
			printf("!!! text require more curve %d !!!\n", text_n_curve);
		}

		if (text_n_vertex >= g_text_ctx->_fill_in.vertex.size()) {
			printf("!!! text require more vertex %d !!!\n", text_n_vertex);
		}
	}

	//
	source_country.reset(new RefugeeAnimation);
	target_country.reset(new RefugeeAnimation);

	source_country->node_name_width.set(country_name_width);
	target_country->node_name_width.set(country_name_width);

	source_country->setNodeNumber((int)name_to_id.size());
	target_country->setNodeNumber((int)name_to_id.size());

	source_country->reserve((int)records_by_year.size());
	target_country->reserve((int)records_by_year.size());
	
	for (int i = 0; i < records_by_year.size(); ++i) {
		printf("Frame: %d/%d\n", i + 1, records_by_year.size());

		auto &records = records_by_year[i];

		//
		std::map<int64_t, Link> source_chord_links;
		std::map<int64_t, Link> target_chord_links;

		//
		for (auto &record : records) {
			if (record.id_0 < record.id_1) {
				auto &source_link = find_or_create(source_chord_links, (((int64_t)record.id_0) << 32) | record.id_1);
				auto &target_link = find_or_create(target_chord_links, (((int64_t)record.id_0) << 32) | record.id_1);
				source_link.size_0 = record.value_0;
				source_link.size_1 = record.value_1;
				target_link.size_0 = record.value_1;
				target_link.size_1 = record.value_0;
			}
			else if (record.id_0 > record.id_1) {
				auto &source_link = find_or_create(source_chord_links, (((int64_t)record.id_1) << 32) | record.id_0);
				auto &target_link = find_or_create(target_chord_links, (((int64_t)record.id_1) << 32) | record.id_0);
				source_link.size_0 = record.value_1;
				source_link.size_1 = record.value_0;
				target_link.size_0 = record.value_0;
				target_link.size_1 = record.value_1;
			}
			else {
				auto &source_link = find_or_create(source_chord_links, (((int64_t)record.id_0) << 32) | record.id_1);
				auto &target_link = find_or_create(target_chord_links, (((int64_t)record.id_0) << 32) | record.id_1);
				source_link.size_0 = record.value_0;
				source_link.size_1 = record.value_1;
				target_link.size_0 = record.value_1;
				target_link.size_1 = record.value_0;
			}
		}

		//
		source_country->addFrame(source_chord_links);
		target_country->addFrame(target_chord_links);
	}

	source_country->genAnimation();
	target_country->genAnimation();

}

#define TIME_PER_YEAR (6000)
//#define GET_ANIMATION_T(time) (time < TIME_PER_YEAR ? 1 : (time % TIME_PER_YEAR) / ((float)TIME_PER_YEAR * .66666f))
__forceinline__ __device__ float GET_ANIMATION_T(int time) {
	float t = (time < TIME_PER_YEAR ? 1 : (time % TIME_PER_YEAR) / ((float)TIME_PER_YEAR * .66666f));
	if (t < 1) {
		return t < 0.5f ? 2 * t* t : 1 - 2 * (1 - t) * (1 - t);
	}
	else {
		return 1;
	}	
}

//float t = time < TIME_PER_YEAR ? 1 : (time % TIME_PER_YEAR) / ((float)TIME_PER_YEAR);

#define LARGE_RADIUS 350.f
#define SMALL_RADIUS 320.f

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_update_year_vertex(
	int n_vertex,
	int frame_id,
	float2 *i_vertex,
	uint32_t *i_path_id,
	float2 *o_vertex
	) {

	auto idx = GET_ID();
	if (idx >= n_vertex) { return; }

	auto path_id = i_path_id[idx];
	if (path_id == frame_id) {
		o_vertex[idx] = i_vertex[idx] + make_float2(1920 / 2 - 128,100);
	}
	else {
		o_vertex[idx] = i_vertex[idx] + make_float2(0, -1000);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_update_curve_vertex_pos(
	int n_curves,

	int vertex_offset,
	int path_offset,
	
	uint32_t *i_curve_vertex_pos,
	uint32_t *i_curve_path_id,

	uint32_t *o_curve_vertex_pos,
	uint32_t *o_curve_path_id
	) {

	auto idx = GET_ID();
	if (idx >= n_curves) { return; }
	o_curve_vertex_pos[idx] = i_curve_vertex_pos[idx] + vertex_offset;
	o_curve_path_id[idx] = i_curve_path_id[idx] + path_offset;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_transform_text_vertex(

	float2 center,

	int n_vertex,
	int path_offset,

	float2 *i_vertex,
	uint32_t *i_vertex_path_id,

	float2 *o_vertex,
	uint32_t *o_vertex_path_id,

	int time,

	float* pf_node_t_0,
	float* pf_node_t_1,
	float* pf_node_size,

	float* node_t_0,
	float* node_t_1,
	float* node_size,

	float* text_width,

	uint32_t *p_color
	) {

	auto vertex_id = GET_ID();
	if (vertex_id >= n_vertex) { return; }

	auto path_id = i_vertex_path_id[vertex_id];
	auto idx = path_id;

	auto nft0 = node_t_0[idx];
	auto nft1 = node_t_1[idx];
	auto nfs = node_size[idx];

	auto pft0 = pf_node_t_0 ? pf_node_t_0[idx] : nft0;
	auto pft1 = pf_node_t_1 ? pf_node_t_1[idx] : nft0;
	auto pfs = pf_node_size ? pf_node_size[idx] : nft0;

	float t = GET_ANIMATION_T(time);
	
	t = max(0.f, t);
	t = min(1.f, t);

	auto t0 = pft0 * (1 - t) + nft0 * t;
	auto t1 = pft1 * (1 - t) + nft1 * t;

	auto text_t = (t0 + t1) * .5f;

	//
	auto vertex = i_vertex[vertex_id];
	vertex.y -= 7;

	float2 translate = make_float2(sin(text_t), cos(text_t)) * (LARGE_RADIUS + 10.f);

	float2 tv;

	const float pi = acos(-1.f);

	if ( nfs == 0.f && pfs == 0.f ) {
		tv = vertex + make_float2(-1000, -1000);
	}
	else {

		if (text_t <= pi) {

			auto tra = -(text_t - acos(-1.f) / 2.f);

			float2 m0 = make_float2(cos(tra), -sin(tra));
			float2 m1 = make_float2(sin(tra), cos(tra));

			tv.x = dot(m0, vertex);
			tv.y = dot(m1, vertex);

			tv = tv + center + translate;

		}
		else {

			auto tra = -(text_t + acos(-1.f) / 2.f);

			float2 m0 = make_float2(cos(tra), -sin(tra));
			float2 m1 = make_float2(sin(tra), cos(tra));

			auto tw = text_width[path_id];
			auto offset = make_float2(-tw, 0.f);

			tv.x = dot(m0, vertex + offset);
			tv.y = dot(m1, vertex + offset);

			tv = tv + center + translate;
		}
		
	}

	if (nfs == 0.f || pfs == 0.f) {

		auto color_t = (pfs * (1 - t) + nfs * t) / max(nfs, pfs);

		color_t = min(1.f, color_t);
		color_t = max(0.f, color_t);

		auto color = p_color[path_offset + path_id];
		color = color & 0x00FFFFFF;
		uint32_t new_alpha = ((int32_t)(color_t * 255.f)) << 24;
		if (new_alpha == 0) { 
			new_alpha = 0x01000000; 
			tv = vertex + make_float2(-1000, -1000);
		}
		color |= new_alpha;

		p_color[path_offset + path_id] = color;

	}

	o_vertex[vertex_id] = tv;
	o_vertex_path_id[vertex_id] = path_id + path_offset;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_evaluate_node(
	int n_nodes,
	int time,

	float* pf_node_t_0,
	float* pf_node_t_1,
	float* pf_node_size,

	float* node_t_0,
	float* node_t_1,
	float* node_size,

	float* animation_t_0,
	float* animation_t_1,
	float* animation_size
	) {

	auto idx = GET_ID();

	if (idx >= n_nodes) { return; }

	auto nft0 = node_t_0[idx];
	auto nft1 = node_t_1[idx];
	auto nfs = node_size[idx];

	auto pft0 = pf_node_t_0 ? pf_node_t_0[idx] : nft0;
	auto pft1 = pf_node_t_1 ? pf_node_t_1[idx] : nft0;
	auto pfs = pf_node_size ? pf_node_size[idx] : nft0;

	float t = GET_ANIMATION_T(time);
	
	t = max(0.f, t);
	t = min(1.f, t);

	animation_t_0[idx] = pft0 * (1 - t) + nft0 * t;
	animation_t_1[idx] = pft1 * (1 - t) + nft1 * t;
	animation_size[idx] = pfs * (1 - t) + nfs * t;

}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_evaluate_link(
	int n_links,
	int time,

	float* pf_link_size_0,
	float* pf_link_size_1,

	float* pf_link_pos_0,
	float* pf_link_pos_1,

	float* link_size_0,
	float* link_size_1,

	float* link_pos_0,
	float* link_pos_1,

	float* animation_link_size_0,
	float* animation_link_size_1,

	float* animation_link_pos_0,
	float* animation_link_pos_1

	) {

	auto idx = GET_ID();

	if (idx >= n_links) { return; }

	auto nfs0 = link_size_0[idx];
	auto nfs1 = link_size_1[idx];

	auto nfp0 = link_pos_0[idx];
	auto nfp1 = link_pos_1[idx];

	auto pfs0 = pf_link_size_0 ? pf_link_size_0[idx] : nfs0;
	auto pfs1 = pf_link_size_1 ? pf_link_size_1[idx] : nfs1;

	auto pfp0 = pf_link_pos_0 ? pf_link_pos_0[idx] : nfp0;
	auto pfp1 = pf_link_pos_1 ? pf_link_pos_1[idx] : nfp1;

	float t = GET_ANIMATION_T(time);
	
	t = max(0.f, t);
	t = min(1.f, t);

	//if (idx == 6) {
	//	printf("(%f %f) (%f %f)\n", pfp0, nfp0, pfp1, nfp1);
	//}

	animation_link_size_0[idx] = pfs0 * (1 - t) + nfs0 * t;
	animation_link_size_1[idx] = pfs1 * (1 - t) + nfs1 * t;

	animation_link_pos_0[idx] = pfp0 * (1 - t) + nfp0 * t;
	animation_link_pos_1[idx] = pfp1 * (1 - t) + nfp1 * t;
}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_gen_node_vg(

	float2 center,

	int n_nodes,

	float *i_node_t0,
	float *i_node_t1,

	int *i_node_color,

	float2 *p_vertex,
	uint32_t *p_vertex_path_id,

	uint32_t *p_curve_vertex_pos,
	uint8_t *p_curve_type,
	uint32_t *p_curve_path_id,
	uint32_t *p_curve_contour_id,
	float *p_curve_arc_w,
	float *p_curve_offset,
	uint8_t *p_curve_reversed,
	uint8_t *p_path_fill_rule,
	uint32_t *p_path_fill_info,

	int vertex_offset,
	int curve_offset,
	int path_offset

	) {

	auto idx = GET_ID();

	if (idx >= n_nodes) { return; }

	//const auto pi = acos(-1.f);

	auto t0 = i_node_t0[idx];
	auto t1 = i_node_t1[idx];

	//printf("%d %f %f\n", idx, t0, t1);

	auto curve_id = curve_offset + idx * 6;
	auto curve_vertex_pos = vertex_offset + idx * 6 * 4;

	//if (t1 - t0 > (pi / 360.f)) {
	//	t0 += pi / 720.f;
	//	t1 -= pi / 720.f;
	//}

	//if (t1 - t0 > (pi / 360.f)) {
	//	float ext = t1 - t0 - pi / 360.f;
	//	float delta = pi / 720.f;
	//	delta = min(delta, ext);
	//	t0 += delta;
	//	t1 -= delta;
	//}

	float2 ip[7];
	float2 op[7];
	float2 dir[7];

	if (t0 == t1) {
		for (int i = 0; i < 7; ++i) {
			ip[i] = make_float2(-1000.f, -1000.f);
			op[i] = make_float2(-1000.f, -1000.f);
		}
	}
	else 
	{
		{
			float t = t0;
			auto d = make_float2(sin(t), cos(t));
			dir[0] = normalize(d);
			ip[0] = center + d * SMALL_RADIUS;
			op[0] = center + d * LARGE_RADIUS;
		}

		{
			float t = t0 * 0.5 + t1 * 0.5;
			auto d = make_float2(sin(t), cos(t));
			dir[3] = normalize(d);
			ip[3] = center + d * SMALL_RADIUS;
			op[3] = center + d * LARGE_RADIUS;
		}

		{
			float t = t1;
			auto d = make_float2(sin(t), cos(t));
			dir[6] = normalize(d);
			ip[6] = center + d * SMALL_RADIUS;
			op[6] = center + d * LARGE_RADIUS;
		}

		// 
		auto in_norm_len = sin((t1 - t0) * .5f) * .353f * SMALL_RADIUS;
		auto out_norm_len = sin((t1 - t0) * .5f) * .353f * LARGE_RADIUS;

		//
		float2 n0, n3, n6;

		n0 = make_float2(-dir[0].y, dir[0].x);
		n3 = make_float2(-dir[3].y, dir[3].x);
		n6 = make_float2(-dir[6].y, dir[6].x);

		ip[1] = ip[0] - n0 * in_norm_len;
		ip[2] = ip[3] + n3 * in_norm_len;
		ip[4] = ip[3] - n3 * in_norm_len;
		ip[5] = ip[6] + n6 * in_norm_len;

		op[1] = op[0] - n0 * out_norm_len;
		op[2] = op[3] + n3 * out_norm_len;
		op[4] = op[3] - n3 * out_norm_len;
		op[5] = op[6] + n6 * out_norm_len;

	}

	auto path_id = path_offset + idx;

	p_path_fill_rule[path_id] = 0;
	p_path_fill_info[path_id] = i_node_color[idx];

	#pragma unroll
	for (int i = 0; i < 6; ++i) {
		p_curve_reversed[curve_id + i] = 0;
		p_curve_path_id[curve_id + i] = path_id;
	}

	#pragma unroll
	for (int i = 0; i < 24; ++i) {
		p_vertex_path_id[curve_vertex_pos + i] = path_id;
	}

	//
	p_curve_type[curve_id + 0] = 2;
	p_curve_vertex_pos[curve_id + 0] = curve_vertex_pos + 0;
	p_vertex[curve_vertex_pos + 0] = op[0];
	p_vertex[curve_vertex_pos + 1] = ip[0];

	p_curve_type[curve_id + 1] = 2;
	p_curve_vertex_pos[curve_id + 1] = curve_vertex_pos + 2;
	p_vertex[curve_vertex_pos + 2] = ip[6];
	p_vertex[curve_vertex_pos + 3] = op[6];

	//
	p_curve_type[curve_id + 2] = 4;
	p_curve_vertex_pos[curve_id + 2] = curve_vertex_pos + 4;
	p_vertex[curve_vertex_pos + 4] = ip[0];
	p_vertex[curve_vertex_pos + 5] = ip[1];
	p_vertex[curve_vertex_pos + 6] = ip[2];
	p_vertex[curve_vertex_pos + 7] = ip[3];

	p_curve_type[curve_id + 3] = 4;
	p_curve_vertex_pos[curve_id + 3] = curve_vertex_pos + 8;
	p_vertex[curve_vertex_pos + 8] = ip[3];
	p_vertex[curve_vertex_pos + 9] = ip[4];
	p_vertex[curve_vertex_pos + 10] = ip[5];
	p_vertex[curve_vertex_pos + 11] = ip[6];

	//
	p_curve_type[curve_id + 4] = 4;
	p_curve_vertex_pos[curve_id + 4] = curve_vertex_pos + 12;
	p_vertex[curve_vertex_pos + 12] = op[6];
	p_vertex[curve_vertex_pos + 13] = op[5];
	p_vertex[curve_vertex_pos + 14] = op[4];
	p_vertex[curve_vertex_pos + 15] = op[3];

	p_curve_type[curve_id + 5] = 4;
	p_curve_vertex_pos[curve_id + 5] = curve_vertex_pos + 16;
	p_vertex[curve_vertex_pos + 16] = op[3];
	p_vertex[curve_vertex_pos + 17] = op[2];
	p_vertex[curve_vertex_pos + 18] = op[1];
	p_vertex[curve_vertex_pos + 19] = op[0];

}

// -------- -------- -------- -------- -------- -------- -------- --------
__global__ void k_gen_link_vg(

	float2 center,

	int n_links,
	
	float *i_link_pos_0,
	float *i_link_pos_1,

	float *i_link_size_0,
	float *i_link_size_1,

	uint32_t *i_link_node_0,
	uint32_t *i_link_node_1,

	int n_nodes,

	float *i_node_size,
	float *i_node_t0,
	float *i_node_t1,
	int *i_node_color,

	float2 *p_vertex,
	uint32_t *p_vertex_path_id,

	uint32_t *p_curve_vertex_pos,
	uint8_t *p_curve_type,
	uint32_t *p_curve_path_id,
	uint32_t *p_curve_contour_id,
	float *p_curve_arc_w,
	float *p_curve_offset,
	uint8_t *p_curve_reversed,
	uint8_t *p_path_fill_rule,
	uint32_t *p_path_fill_info,

	int vertex_offset,
	int curve_offset,
	int path_offset

	) {

	auto idx = GET_ID();
	if (idx >= n_links) { return; }

	auto pos_0 = i_link_pos_0[idx];
	auto pos_1 = i_link_pos_1[idx];

	auto size_0 = i_link_size_0[idx];
	auto size_1 = i_link_size_1[idx];

	auto node_0 = i_link_node_0[idx];
	auto node_1 = i_link_node_1[idx];

	auto node_0_t0 = i_node_t0[node_0];
	auto node_0_t1 = i_node_t1[node_0];

	auto node_1_t0 = i_node_t0[node_1];
	auto node_1_t1 = i_node_t1[node_1];

	auto node_size_0 = i_node_size[node_0];
	auto node_size_1 = i_node_size[node_1];

	//
	const auto pi = acos(-1.f);

	//
	auto n0_t0 = node_0_t0;
	auto n0_t1 = node_0_t1;

	auto n1_t0 = node_1_t0;
	auto n1_t1 = node_1_t1;

	//
	auto link_t_00 = pos_0 / node_size_0;
	auto link_t_01 = (pos_0 + size_0) / node_size_0;

	auto link_t_10 = pos_1 / node_size_1;
	auto link_t_11 = (pos_1 + size_1) / node_size_1;

	//
	auto llerp = [](float a, float b, float t) {
		return a*(1 - t) + b*t;
	};

	link_t_00 = llerp(n0_t0, n0_t1, link_t_00);
	link_t_01 = llerp(n0_t0, n0_t1, link_t_01);

	link_t_10 = llerp(n1_t0, n1_t1, link_t_10);
	link_t_11 = llerp(n1_t0, n1_t1, link_t_11);

	//float t;

	float2 ip_0[7];
	float2 dir_0[7];

	{

		{
			float t = link_t_00;
			auto d = make_float2(sin(t), cos(t));
			dir_0[0] = normalize(d);
			ip_0[0] = center + d * SMALL_RADIUS;
		}

		{
			float t = link_t_00 * 0.5 + link_t_01 * 0.5;
			auto d = make_float2(sin(t), cos(t));
			dir_0[3] = normalize(d);
			ip_0[3] = center + d * SMALL_RADIUS;
		}

		{
			float t = link_t_01;
			auto d = make_float2(sin(t), cos(t));
			dir_0[6] = normalize(d);
			ip_0[6] = center + d * SMALL_RADIUS;
		}

		// 
		auto in_norm_len = sin((link_t_01 - link_t_00) * .5f) * .353f * SMALL_RADIUS;
		auto out_norm_len = sin((link_t_01 - link_t_00) * .5f) * .353f * LARGE_RADIUS;

		//
		float2 n0, n3, n6;

		n0 = make_float2(-dir_0[0].y, dir_0[0].x);
		n3 = make_float2(-dir_0[3].y, dir_0[3].x);
		n6 = make_float2(-dir_0[6].y, dir_0[6].x);

		ip_0[1] = ip_0[0] - n0 * in_norm_len;
		ip_0[2] = ip_0[3] + n3 * in_norm_len;
		ip_0[4] = ip_0[3] - n3 * in_norm_len;
		ip_0[5] = ip_0[6] + n6 * in_norm_len;

	}

	float2 ip_1[7];
	float2 dir_1[7];

	{

		{
			float t = link_t_10;
			auto d = make_float2(sin(t), cos(t));
			dir_1[0] = normalize(d);
			ip_1[0] = center + d * SMALL_RADIUS;
		}

		{
			float t = link_t_10 * 0.5 + link_t_11 * 0.5;
			auto d = make_float2(sin(t), cos(t));
			dir_1[3] = normalize(d);
			ip_1[3] = center + d * SMALL_RADIUS;
		}

		{
			float t = link_t_11;
			auto d = make_float2(sin(t), cos(t));
			dir_1[6] = normalize(d);
			ip_1[6] = center + d * SMALL_RADIUS;
		}

		// 
		auto in_norm_len = sin((link_t_11 - link_t_10) * .5f) * .353f * SMALL_RADIUS;
		auto out_norm_len = sin((link_t_11 - link_t_10) * .5f) * .353f * LARGE_RADIUS;

		//
		float2 n0, n3, n6;

		n0 = make_float2(-dir_1[0].y, dir_1[0].x);
		n3 = make_float2(-dir_1[3].y, dir_1[3].x);
		n6 = make_float2(-dir_1[6].y, dir_1[6].x);

		ip_1[1] = ip_1[0] - n0 * in_norm_len;
		ip_1[2] = ip_1[3] + n3 * in_norm_len;
		ip_1[4] = ip_1[3] - n3 * in_norm_len;
		ip_1[5] = ip_1[6] + n6 * in_norm_len;

	}

	float theta_small = abs(link_t_10 - link_t_01);
	float theta_large = abs(link_t_11 - link_t_00);

	if (theta_small > pi) { theta_small -= pi; }
	if (theta_large > pi) { theta_large -= pi; }

	theta_small *= .5f;
	theta_large *= .5f;

	auto r_small = SMALL_RADIUS / (2 * sin(theta_small) + 1.f);
	auto r_large = SMALL_RADIUS / (2 * sin(theta_large) + 1.f);

	auto c00 = center + dir_0[0] * r_large;
	auto c01 = center + dir_0[6] * r_small;
	auto c10 = center + dir_1[0] * r_small;
	auto c11 = center + dir_1[6] * r_large;

	auto curve_id = curve_offset + idx * 6;
	auto curve_vertex_pos = vertex_offset + idx * 6 * 4;
	auto path_id = path_offset + idx;

	p_path_fill_rule[path_id] = 0;
	p_path_fill_info[path_id] =
		(i_node_color[size_0 > size_1 ? node_0 : node_1] & 0x7FFFFFFF);
		//(i_node_color[size_0 > size_1 ? node_0 : node_1] & 0x00FFFFFF) | (223 << 24);

	#pragma unroll
	for (int i = 0; i < 6; ++i) {
		p_curve_reversed[curve_id + i] = 0;
		p_curve_path_id[curve_id + i] = path_id;
	}

	#pragma unroll
	for (int i = 0; i < 24; ++i) {
		p_vertex_path_id[curve_vertex_pos + i] = path_id;
	}

	auto v00 = ip_0[0];
	auto v01 = ip_0[6];
	auto v10 = ip_1[0];
	auto v11 = ip_1[6];

	//
	p_curve_type[curve_id +0] = 4;
	p_curve_vertex_pos[curve_id + 0] = curve_vertex_pos + 0;
	p_vertex[curve_vertex_pos + 0] = v01;
	p_vertex[curve_vertex_pos + 1] = c01;
	p_vertex[curve_vertex_pos + 2] = c10;
	p_vertex[curve_vertex_pos + 3] = v10;

	p_curve_type[curve_id + 1] = 4;
	p_curve_vertex_pos[curve_id + 1] = curve_vertex_pos + 4;
	p_vertex[curve_vertex_pos + 4] = v11;
	p_vertex[curve_vertex_pos + 5] = c11;
	p_vertex[curve_vertex_pos + 6] = c00;
	p_vertex[curve_vertex_pos + 7] = v00;

	//
	p_curve_type[curve_id + 2] = 4;
	p_curve_vertex_pos[curve_id + 2] = curve_vertex_pos + 8;
	p_vertex[curve_vertex_pos + 8] = ip_0[0];
	p_vertex[curve_vertex_pos + 9] = ip_0[1];
	p_vertex[curve_vertex_pos + 10] = ip_0[2];
	p_vertex[curve_vertex_pos + 11] = ip_0[3];

	p_curve_type[curve_id + 3] = 4;
	p_curve_vertex_pos[curve_id + 3] = curve_vertex_pos + 12;
	p_vertex[curve_vertex_pos + 12] = ip_0[3];
	p_vertex[curve_vertex_pos + 13] = ip_0[4];
	p_vertex[curve_vertex_pos + 14] = ip_0[5];
	p_vertex[curve_vertex_pos + 15] = ip_0[6];

	//
	p_curve_type[curve_id + 4] = 4;
	p_curve_vertex_pos[curve_id + 4] = curve_vertex_pos + 16;
	p_vertex[curve_vertex_pos + 16] = ip_1[0];
	p_vertex[curve_vertex_pos + 17] = ip_1[1];
	p_vertex[curve_vertex_pos + 18] = ip_1[2];
	p_vertex[curve_vertex_pos + 19] = ip_1[3];
	
	p_curve_type[curve_id +5] = 4;
	p_curve_vertex_pos[curve_id + 5] = curve_vertex_pos + 20;
	p_vertex[curve_vertex_pos + 20] = ip_1[3];
	p_vertex[curve_vertex_pos + 21] = ip_1[4];
	p_vertex[curve_vertex_pos + 22] = ip_1[5];
	p_vertex[curve_vertex_pos + 23] = ip_1[6];

}

} // end of Chord Animation

void chord_animation(
	int last_frame_timestamp,
	int next_frame_timestamp,
	RasterizerBase::VGInputCurveDataPack &_last_frame_curve_in,
	RasterizerBase::VGInputCurveDataPack &_next_frame_curve_in,
	RasterizerBase::VGInputPathDataPack &_last_frame_path_in,
	RasterizerBase::VGInputPathDataPack &_next_frame_path_in
	) {

	using namespace ChordAnimation;

	// -------- -------- -------- --------
	static std::shared_ptr<RefugeeAnimation> source_country_animation;
	static std::shared_ptr<RefugeeAnimation> target_country_animation;

	static std::shared_ptr<RefugeeFrameDrawData> animation_frame;

	// -------- -------- -------- --------
	if (!source_country_animation) {
		load_refugee_animation(source_country_animation, target_country_animation);
		animation_frame.reset(new RefugeeFrameDrawData);
		animation_frame->reserve(4000, 100000);
		printf("Animation generated: %d\n", clock());
	}

	// -------- -------- -------- --------
	int frame_id = (next_frame_timestamp / TIME_PER_YEAR) + 1;
	//frame_id = (frame_id % 2) + 1;
	//int frame_id = 1;

	// -------- -------- -------- --------
	auto &_vg = _next_frame_curve_in;
	auto &_path = _next_frame_path_in;

	_vg.n_curves = 0;
	_vg.n_vertices = 0;
	_path.n_paths = 0;

	// -------- -------- -------- --------
	// add the years

	{

		auto &year_ctx = *g_year_text_ctx;
		auto year_vertex_number = year_ctx._nextVertexIndex;
		auto year_curve_number = year_ctx._nextCurveIndex;
		auto year_path_number = number_of_years;

		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("0");

		// year path
		cudaMemcpyAsync(_path.fill_info.gptr(),
			g_year_colors.data(),
			year_path_number * 4,
			cudaMemcpyHostToDevice);

		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("1");

		// year curve
		cudaMemcpyAsync(_vg.curve_type.gptr(),
			year_ctx._fill_in.curve_type.gptr(),
			year_curve_number,
			cudaMemcpyDeviceToDevice);

		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("2");

		cudaMemcpyAsync(_vg.curve_vertex_pos.gptr(),
			year_ctx._fill_in.curve_vertex_pos.gptr(),
			year_curve_number * 4,
			cudaMemcpyDeviceToDevice);

		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("3");

		cudaMemcpyAsync(_vg.curve_path_id.gptr(),
			year_ctx._fill_in.curve_path_id.gptr(),
			year_curve_number * 4,
			cudaMemcpyDeviceToDevice);

		// year vertex

		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("before update year vertex");

		k_update_year_vertex << < year_vertex_number, 256 >> > (

			year_vertex_number,
			frame_id,

			year_ctx._fill_in.vertex.gptr(),
			year_ctx._fill_in.vertex_path_id.gptr(),

			_vg.vertex.gptr());

		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("after update year vertex");

		//cudaMemcpyAsync(_vg.vertex.gptr(),
		//	year_ctx._fill_in.vertex.gptr(),
		//	year_vertex_number * sizeof(float2),
		//	cudaMemcpyDeviceToDevice);

		cudaMemcpyAsync(_vg.vertex_path_id.gptr(),
			year_ctx._fill_in.vertex_path_id.gptr(),
			year_vertex_number * 4,
			cudaMemcpyDeviceToDevice);

		
		_path.n_paths = year_path_number;
		_vg.n_curves = year_curve_number;
		_vg.n_vertices = year_vertex_number;

	}

	// -------- -------- -------- --------
	auto addChord = [frame_id, next_frame_timestamp](
		RasterizerBase::VGInputCurveDataPack &_vg,
		RasterizerBase::VGInputPathDataPack &_path,
		std::shared_ptr<RefugeeAnimation> p_animation, float2 center) {

		auto &current_frame = p_animation->frame(frame_id);
		auto &frame_t0 = current_frame.t0;
		auto &frame_t1 = current_frame.t1;

		auto &aframe = *animation_frame;

		int n_nodes = ((int)frame_t0.node_size.size());

		k_evaluate_node << <n_nodes, 256 >> >(
			n_nodes,
			next_frame_timestamp,
			frame_t0.node_t_0, frame_t0.node_t_1, frame_t0.node_size,
			frame_t1.node_t_0, frame_t1.node_t_1, frame_t1.node_size,
			aframe.node_t_0, aframe.node_t_1, aframe.node_size
			);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("evaluate_node");

		int n_links = (int)frame_t0.link_size_0.size();
		k_evaluate_link << < n_links, 256 >> > (
			n_links,
			next_frame_timestamp,
			frame_t0.link_size_0, frame_t0.link_size_1, frame_t0.link_pos_0, frame_t0.link_pos_1,
			frame_t1.link_size_0, frame_t1.link_size_1, frame_t1.link_pos_0, frame_t1.link_pos_1,
			aframe.link_size_0, aframe.link_size_1, aframe.link_pos_0, aframe.link_pos_1
			);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("evaluate_link");

#define ENABLE_TEXT
#ifdef ENABLE_TEXT

		auto text_n_curve = g_text_ctx->_nextCurveIndex;
		auto text_n_vertex = g_text_ctx->_nextVertexIndex;
		//auto text_n_paths = n_nodes;

		auto &text_ctx = *g_text_ctx;

		cudaMemcpyAsync(_vg.curve_type.gptr() + _vg.n_curves,
			g_text_ctx->_fill_in.curve_type.gptr(),
			text_n_curve,
			cudaMemcpyDeviceToDevice);

		k_update_curve_vertex_pos<<<text_n_curve, 1024>>>(
			text_n_curve,

			_vg.n_vertices,
			_path.n_paths,

			text_ctx._fill_in.curve_vertex_pos.gptr(),
			text_ctx._fill_in.curve_path_id.gptr(),

			_vg.curve_vertex_pos.gptr() + _vg.n_curves,
			_vg.curve_path_id.gptr() + _vg.n_curves

			);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("update curve vertex pos");

		cudaMemcpyAsync(_path.fill_info.gptr() + _path.n_paths,
			frame_t0.node_color.gptr() ,
			4 * n_nodes,
			cudaMemcpyDeviceToDevice);

		k_transform_text_vertex << <  text_n_vertex, 256 >> > (

			center,

			text_n_vertex,
			_path.n_paths,

			text_ctx._fill_in.vertex.gptr(),
			text_ctx._fill_in.vertex_path_id.gptr(),

			_vg.vertex.gptr() + _vg.n_vertices,
			_vg.vertex_path_id.gptr() + _vg.n_vertices,
			
			next_frame_timestamp,
			frame_t0.node_t_0, frame_t0.node_t_1, frame_t0.node_size,
			frame_t1.node_t_0, frame_t1.node_t_1, frame_t1.node_size,

			p_animation->node_name_width.gptr(),

			_path.fill_info.gptr()

			);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("transform text vertex");

		_path.n_paths += n_nodes;
		_vg.n_curves += text_n_curve;
		_vg.n_vertices += text_n_vertex;

#endif

#define ENABLE_NODE
#ifdef ENABLE_NODE

		k_gen_node_vg << < n_nodes, 256 >> > (

			center,

			n_nodes,

			aframe.node_t_0,
			aframe.node_t_1,

			frame_t0.node_color,

			_vg.vertex,
			_vg.vertex_path_id,

			_vg.curve_vertex_pos,
			_vg.curve_type,
			_vg.curve_path_id,
			_vg.curve_contour_id,
			_vg.curve_arc_w,
			_vg.curve_offset,
			_vg.curve_reversed,

			_path.fill_rule,
			_path.fill_info,

			_vg.n_vertices,
			_vg.n_curves,
			_path.n_paths
			);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_node_vg");

		_path.n_paths += n_nodes;
		_vg.n_curves += n_nodes * 6;
		_vg.n_vertices += n_nodes * 6 * 4;

#endif

#define ENABLE_LINK
#ifdef ENABLE_LINK
		k_gen_link_vg << < n_links, 256 >> > (

			center,

			n_links,

			aframe.link_pos_0,
			aframe.link_pos_1,

			aframe.link_size_0,
			aframe.link_size_1,

			frame_t0.link_node_0,
			frame_t0.link_node_1,

			n_nodes,

			aframe.node_size,
			aframe.node_t_0,
			aframe.node_t_1,

			frame_t0.node_color,

			_vg.vertex,
			_vg.vertex_path_id,

			_vg.curve_vertex_pos,
			_vg.curve_type,
			_vg.curve_path_id,
			_vg.curve_contour_id,
			_vg.curve_arc_w,
			_vg.curve_offset,
			_vg.curve_reversed,

			_path.fill_rule,
			_path.fill_info,

			_vg.n_vertices,
			_vg.n_curves,
			_path.n_paths
			);
		DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR("k_gen_link_vg");

		_path.n_paths += n_links;
		_vg.n_curves += n_links * 6;
		_vg.n_vertices += n_links * 6 * 4;

#endif

	};

	// -------- -------- -------- --------
	addChord(_vg, _path, source_country_animation, make_float2(480, 540));
	addChord(_vg, _path, target_country_animation, make_float2(1440, 540));

	// check reserved space.
	if (_path.n_paths >= _next_frame_path_in.fill_rule.reserved()) {
		printf("%d %d\n", _path.n_paths, _next_frame_path_in.fill_rule.reserved());
		throw std::runtime_error("vg_animation: reserve more path");
	}

	if (_vg.n_curves >= _next_frame_curve_in.curve_type.reserved()) {
		printf("%d %d\n", _vg.n_curves, _next_frame_curve_in.curve_type.reserved());
		throw std::runtime_error("vg_animation: reserve more curve");
	}

	if (_vg.n_vertices >= _next_frame_curve_in.vertex.reserved()) {
		printf("%d %d\n", _vg.n_vertices, _next_frame_curve_in.vertex.reserved());
		throw std::runtime_error("vg_animation: reserve more vertex");
	}


	
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

//using namespace ChordAnimation;

void vg_animation(
	int last_frame_timestamp,
	int next_frame_timestamp,
	RasterizerBase::VGInputCurveDataPack &_last_frame_curve_in,
	RasterizerBase::VGInputCurveDataPack &_next_frame_curve_in,
	RasterizerBase::VGInputPathDataPack &_last_frame_path_in,
	RasterizerBase::VGInputPathDataPack &_next_frame_path_in
	) {

	//tree_animation(last_frame_timestamp, next_frame_timestamp,
	//	_last_frame_curve_in, _next_frame_curve_in,
	//	_last_frame_path_in, _next_frame_path_in);

	chord_animation(last_frame_timestamp, next_frame_timestamp,
		_last_frame_curve_in, _next_frame_curve_in,
		_last_frame_path_in, _next_frame_path_in);

}

}
