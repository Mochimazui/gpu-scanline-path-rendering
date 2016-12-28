
#include "ras_base.h"
#include "ras_define.h"

#include <ctime>
#include <cstdlib>

#include <functional>
#include <algorithm>

#include "mochimazui/3rd/helper_cuda.h"
#include "mochimazui/stdio_ext.h"
#include <mochimazui/glpp.h>

#include "cuda/cuda_cached_allocator.h"

#include <mochimazui/3rd/stb_image.h>
#include <mochimazui/3rd/stb_image_resize.h>
#include <mochimazui/3rd/stb_image_write.h>

//#include <mochimazui/3rd/stb_truetype.c>

#include "../../vg_container.h"
#include "../../cutil_math.h"

#define NO_GL_EXT

namespace Mochimazui {

namespace RasterizerBase {

using stdext::error_printf;

// -------- -------- -------- -------- -------- -------- -------- --------
VGRasterizer::VGRasterizer() {

	_input_transform[0] = make_float4(1.f, 0.f, 0.f, 0.f);
	_input_transform[0] = make_float4(0.f, 1.f, 0.f, 0.f);
	_input_transform[0] = make_float4(0.f, 0.f, 1.f, 0.f);
	_input_transform[0] = make_float4(0.f, 0.f, 0.f, 1.f);

	_output_transform[0] = make_float4(1.f, 0.f, 0.f, 0.f);
	_output_transform[0] = make_float4(0.f, 1.f, 0.f, 0.f);
	_output_transform[0] = make_float4(0.f, 0.f, 1.f, 0.f);
	_output_transform[0] = make_float4(0.f, 0.f, 0.f, 1.f);

}

VGRasterizer::~VGRasterizer() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::init() {
	initProgram();
	initTextureAndFramebuffer();
	_base_gl_util.init();

	// reserve space for animation
	if (_animation) {

		//int n = 524288; // for tree 0x1FFFF
		int n = 983035; // for tree 0x2FFFF

		_vg_in_curve[1].reserve(n, n * 4);
		_vg_in_path[1].reserve(n, 0);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::addVg(const VGContainer &vg) {

	_vg_size.x = vg.width;
	_vg_size.y = vg.height;

	using std::vector;

	// -------- -------- -------- --------
	// bind old input pack
	auto &_raw_in = _vg_in_curve[0];
	auto &_fill_in = _vg_in_curve[1];
	auto &_path_in = _vg_in_path[0];

	// -------- -------- -------- --------
	auto &_in = _gpu_stroke_to_fill ? _raw_in : _fill_in;

	_path_in.n_paths = vg.pathNumber();
	_in.n_curves = vg.curveNumber();
	_in.n_vertices = vg.vertexNumber();

	//vector<float2> vertex;
	vector<uint32_t> vertex_path_id;

	vector<uint32_t> curve_vertex_pos;
	vector<uint32_t> curve_path_id;
	vector<uint8_t> curve_type;
	vector<uint8_t> curve_reversed;
	vector<float> curve_arc_w;
	vector<float> curve_offset;
	vector<uint32_t> curve_contour_id;

	vector<uint32_t> contour_first_curve;
	vector<uint32_t> contour_last_curve;

	vector<uint8_t> fill_rule;
	vector<uint32_t> fill_info;

	vector<uint32_t> stroke_info;
	vector<float> stroke_width;
	vector<uint8_t> stroke_cap_and_join;
	vector<float> stroke_miter_limit;

	vector<uint32_t> curve_index;

	vector<glm::vec2> vertex;

	// reserve
	vertex_path_id.reserve(_path_in.n_paths);
	fill_rule.reserve(_path_in.n_paths);
	fill_info.reserve(_path_in.n_paths);

	stroke_info.reserve(_path_in.n_paths);
	stroke_width.reserve(_path_in.n_paths);
	stroke_cap_and_join.reserve(_path_in.n_paths);
	stroke_miter_limit.reserve(_path_in.n_paths);

	contour_first_curve.reserve(_path_in.n_contours);
	contour_last_curve.reserve(_path_in.n_contours);

	curve_vertex_pos.reserve(_in.n_curves);
	curve_path_id.reserve(_in.n_curves);
	curve_type.reserve(_in.n_curves);
	curve_reversed.reserve(_in.n_curves);
	curve_arc_w.reserve(_in.n_curves);
	curve_offset.reserve(_in.n_curves);
	curve_contour_id.reserve(_in.n_curves);

	vertex.reserve(_in.n_vertices);

	long long timestamp_0;
	QueryPerformanceCounter((LARGE_INTEGER*)&timestamp_0);
	
	//
	for (uint32_t ip = 0; ip < vg.pathNumber(); ++ip) {

		auto path_id = ip;

		auto fill_type = vg.path.fillType[ip];
		if (fill_type == FT_COLOR) {
			uint32_t c = (uint32_t)vg.path.fillColor[ip];
			c = c & 0xFF000000 ? c : 0;
			fill_info.push_back(c);
		}
		else if (fill_type == FT_GRADIENT) {
			fill_info.push_back(path_id + 1);
		}
		else {
			fill_info.push_back(0);
		}
		fill_rule.push_back(vg.path.fillRule[ip]);

		auto stroke_type = vg.path.strokeType[ip];
		if (stroke_type == ST_COLOR) {
			uint32_t c = (uint32_t)vg.path.strokeColor[ip];
			c = c & 0xFF000000 ? c : 0;
			stroke_info.push_back(c);
			fill_info.back() = c;
		}
		else {
			stroke_info.push_back(0);
		}
		stroke_width.push_back(vg.path.strokeWidth[ip]);
		stroke_cap_and_join.push_back(
			(vg.path.strokeLineCap[ip] << 4)
			|
			(vg.path.strokeLineJoin[ip])
			);

		stroke_miter_limit.push_back(vg.path.strokeMiterLimit[ip]);

		for (uint32_t icon = 0; icon < vg.path.contourNumber[ip]; ++icon) {
			auto con_offset = vg.path.contourIndex[ip];
			auto con_id = con_offset + icon;

			uint32_t contour_first_vertex = (uint32_t)(vg.vertex.point.size() + 1);
			uint32_t contour_last_vertex = 0;

			contour_first_curve.push_back(vg.contour.curveIndex[con_id]);
			contour_last_curve.push_back(vg.contour.curveIndex[con_id] + vg.contour.curveNumber[con_id] - 1);
			auto contour_closed = vg.contour.closed[con_id];

			for (uint32_t icurve = 0; icurve < vg.contour.curveNumber[con_id]; ++icurve) {

				auto curve_begin = vg.contour.curveIndex[con_id];
				auto curve_id = curve_begin + icurve;

				auto curve_vertex_index = vg.curve.vertexIndex[curve_id];
				curve_vertex_pos.push_back(curve_vertex_index);

				if (_gpu_stroke_to_fill) {
					uint32_t first_flag = icurve == 0 ? 0x40000000 : 0;
					uint32_t last_flag = (icurve+1) == vg.contour.curveNumber[con_id] ? 0x20000000 : 0;
					curve_path_id.push_back(path_id | (contour_closed ? 0x80000000 : 0)
						| first_flag | last_flag );
				}
				else {
					curve_path_id.push_back(path_id);
				}

				auto type = vg.curve.type[curve_id];
				auto offset = vg.curve.offset[curve_id];
				curve_type.push_back(type);
				curve_offset.push_back(offset);
				curve_contour_id.push_back(con_id);

				contour_first_vertex = std::min(contour_first_vertex, curve_vertex_index);
				contour_last_vertex = std::max(contour_last_vertex,
					curve_vertex_index + (type & 7) - 1);

				switch (type) {
				default:
					assert(0);
					break;
				case CT_Linear:
					curve_arc_w.push_back(-1.f);
					break;
				case CT_Quadratic:
					curve_arc_w.push_back(-1.f);					
					break;
				case CT_Cubic:
					curve_arc_w.push_back(-1.f);					
					break;
				case CT_Rational:
				{
					auto cw = vg.curve.arc_w1s[curve_id];
					curve_arc_w.push_back(*((float*)&cw));
					break;
				}
				}

				for (int i = 0; i < (type & 7); ++i) {
					vertex_path_id.push_back(path_id);
				}

			} // end of CONTOUR loop.

		}	
	}

	vertex = vg.vertex.point;
	curve_reversed.resize(vg.curveNumber() + _reservedCurveNumber);
	memset(curve_reversed.data(), 0, sizeof(uint8_t) * (vg.curveNumber() + _reservedCurveNumber));

	_nextPathIndex = vg.pathNumber();
	_nextCurveIndex = vg.curveNumber();
	_nextVertexIndex = vg.vertexNumber();

	for (int i = 0; i < _reservedPathNumber; ++i) {
		auto pathIndex = i + vg.pathNumber();
		fill_rule.push_back(0);
		fill_info.push_back(0xFF00FF00);
	}

	for (int i = 0; i < _reservedCurveNumber; ++i) {
		curve_vertex_pos.push_back(0);
		curve_path_id.push_back(0);
		curve_type.push_back(CT_Linear);
		curve_arc_w.push_back(0);
		curve_reversed.push_back(0);
	}

	for (int i = 0; i < _reservedVertexNumber; ++i) {
		vertex.push_back(glm::vec2(-1, -1));
		vertex.push_back(glm::vec2(-1, -1));
		vertex_path_id.push_back(0);
		vertex_path_id.push_back(0);
	}

	long long timestamp_1;
	QueryPerformanceCounter((LARGE_INTEGER*)&timestamp_1);

	// copy vg data
	_in.vertex.resizeWithoutCopy(vertex.size());
	_in.vertex_path_id.resizeWithoutCopy(vertex_path_id.size());

	//
	_in.curve_vertex_pos.resizeWithoutCopy(curve_vertex_pos.size());
	_in.curve_type.resizeWithoutCopy(curve_type.size());
	_in.curve_path_id.resizeWithoutCopy(curve_path_id.size());
	_in.curve_arc_w.resizeWithoutCopy(curve_arc_w.size());
	_in.curve_offset.resizeWithoutCopy(curve_offset.size());
	_in.curve_reversed.resizeWithoutCopy(curve_reversed.size());
	_in.curve_contour_id.resizeWithoutCopy(curve_contour_id.size());

	//
	_path_in.fill_info.resizeWithoutCopy(fill_info.size());
	_path_in.fill_rule.resizeWithoutCopy(fill_rule.size());

	_path_in.stroke_info.resizeWithoutCopy(stroke_info.size());
	_path_in.stroke_width.resizeWithoutCopy(stroke_width.size());
	_path_in.stroke_cap_join.resizeWithoutCopy(stroke_cap_and_join.size());
	_path_in.stroke_miter_limit.resizeWithoutCopy(stroke_miter_limit.size());

	_path_in.contour_first_curve.resizeWithoutCopy(contour_first_curve.size());
	_path_in.contour_last_curve.resizeWithoutCopy(contour_last_curve.size());

	long long timestamp_2;
	QueryPerformanceCounter((LARGE_INTEGER*)&timestamp_2);
	
	cudaMemcpyAsync(_in.vertex.gptr(), vertex.data(), sizeof(float2) * vertex.size(),
		cudaMemcpyHostToDevice);
	_in.vertex_path_id.setAsync(vertex_path_id);

	//
	_in.curve_vertex_pos.setAsync(curve_vertex_pos);
	_in.curve_type.setAsync(curve_type);
	_in.curve_path_id.setAsync(curve_path_id);
	_in.curve_arc_w.setAsync(curve_arc_w);
	_in.curve_offset.setAsync(curve_offset);
	_in.curve_reversed.setAsync(curve_reversed);
	_in.curve_contour_id.setAsync(curve_contour_id);

	//
	_path_in.fill_info.setAsync(fill_info);
	_path_in.fill_rule.setAsync(fill_rule);

	_path_in.stroke_info.setAsync(stroke_info);
	_path_in.stroke_width.setAsync(stroke_width);
	_path_in.stroke_cap_join.setAsync(stroke_cap_and_join);
	_path_in.stroke_miter_limit.setAsync(stroke_miter_limit);

	_path_in.contour_first_curve.setAsync(contour_first_curve);
	_path_in.contour_last_curve.setAsync(contour_last_curve);

	cudaDeviceSynchronize();

	long long timestamp_3;
	QueryPerformanceCounter((LARGE_INTEGER*)&timestamp_3);

	// update gradient-related data.
	_gradient_ramp = vg.gradient.gradient_ramp_texture;
	_gradient_table = vg.gradient.gradient_table_texture;

	// gradient data to texture

	if (_gradient_ramp.size()) {
		int htex = (int)_gradient_ramp.size() >> 10;
		_base_gl_texture_gradient_ramp.target(GLPP::Texture2D).create();
		_base_gl_texture_gradient_ramp.storage2D(
			1, _enableSRGBCorrection ? GL_SRGB8_ALPHA8 : GL_RGBA8, 1024, htex > 0 ? htex : 1);
			
		_base_gl_texture_gradient_ramp.subImage2D(0, 0, 0, 1024, htex > 0 ? htex : 1,
			GL_RGBA, GL_UNSIGNED_BYTE, htex > 0 ? (void*)&_gradient_ramp[0] : NULL);

		_base_gl_texture_gradient_ramp
			.parameterf(GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			.parameterf(GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			.parameterf(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
			.parameterf(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		_gradient_irampheight = (htex>0 ? 1.f / (float)htex : 0.f);
	}

	if (_gradient_table.size()) {
		_base_gl_buffer_gradient_table.create();
		_base_gl_buffer_gradient_table.storage((GLsizei)_gradient_table.size()*sizeof(int),
			&_gradient_table[0], GL_MAP_READ_BIT);

		_base_gl_texture_gradient_table.target(GLPP::TextureBuffer).create();
		_base_gl_texture_gradient_table.buffer(GL_RGBA32F, _base_gl_buffer_gradient_table);
		
	}

	glFinish();

	long long frequency;
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);

	printf("add vg time: %f %f \n", 
		((double)(timestamp_1 - timestamp_0)) / (double)frequency * 1000.0,
		((double)(timestamp_3 - timestamp_2)) / (double)frequency * 1000.0
		);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::viewport(int x, int y, int width, int height) {
	_vp_pos = glm::ivec2(x, y);
	resizeViewport(width, height);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::resizeViewport(int width, int height) {
	if (_vp_size.x == width && _vp_size.y == height) { return; }
	_vp_size.x = width; _vp_size.y = height;
}

void VGRasterizer::resizeOutputBuffer(int new_width, int new_height) {
	if ((_output_buffer_size.x == new_width) && (_output_buffer_size.y == new_height)) {
		return;
	}
	_output_buffer_size = glm::ivec2(new_width, new_height);
	onResize(new_width, new_height);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::initProgram() {
	_base_gl_program_output_scale.create()
		.createShaderFromFile(GLPP::Vertex, "./shader/shared/output_scale.vert.glsl")
		.createShaderFromFile(GLPP::Fragment, "./shader/shared/output_scale.frag.glsl")
		.link();

	_base_gl_program_curve.create()
		.createShaderFromFile(GLPP::Vertex, "./shader/shared/curve.vert.glsl")
		.createShaderFromFile(GLPP::Fragment, "./shader/shared/curve.frag.glsl")
		.link();

	_base_gl_program_fps.create()
		.createShaderFromFile(GLPP::Vertex, "./shader/shared/fps.vert.glsl")
		.createShaderFromFile(GLPP::Fragment, "./shader/shared/fps.frag.glsl")
		.link();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::initTextureAndFramebuffer() {

	CHECK_GL_ERROR();

	// output
	_base_gl_texture_output.target(GLPP::Texture2DMultisample).create()
		.storage2DMultisample(_multisample_output ? _samples : 1,
			_enableSRGBCorrection ? GL_SRGB8_ALPHA8 : GL_RGBA8, 1024, 1024, GL_TRUE);
	_base_gl_framebuffer_output.create().texture2D(
		GL_COLOR_ATTACHMENT0, _base_gl_texture_output, 0);
	if (_enable_stencil_buffer) {
		_base_gl_texture_output_stencil.target(GLPP::Texture2DMultisample).create()
			.storage2DMultisample(_multisample_output ? _samples : 1, 
				GL_DEPTH24_STENCIL8, 1024, 1024, GL_TRUE);
		_base_gl_framebuffer_output.texture2D(GL_DEPTH_STENCIL_ATTACHMENT,
			_base_gl_texture_output_stencil, 0);
	}
	CHECK_GL_ERROR();

	// output scale
	_base_gl_texture_output_scale.target(GLPP::Texture2D).create()
		.storage2D(1, GL_RGBA8, 1024, 1024);
	_base_gl_texture_output_scale.parameterf(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	_base_gl_texture_output_scale.parameterf(GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		.parameterf(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		.parameterf(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	_base_gl_framebuffer_output_scale.create().texture2D(GL_COLOR_ATTACHMENT0,
		_base_gl_texture_output_scale, 0);
	CHECK_GL_ERROR();

	// output integral sample
	_base_gl_texture_output_integrate_samples.target(GLPP::Texture2D).create()
		.storage2D(1, 
			_enableSRGBCorrection ? GL_SRGB8_ALPHA8 : GL_RGBA8, 
			1024, 1024);
	_base_gl_framebuffer_output_integrate_samples.create().texture2D(GL_COLOR_ATTACHMENT0,
		_base_gl_texture_output_integrate_samples, 0);

	// empty vertex array
	_base_gl_vertex_array_empty.create();

	CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::onResize(int width, int height) {

	// -------- -------- -------- --------
	// update shader uniform.
	_base_gl_program_output_scale.uniform2i("vp_size", width, height);
	_base_gl_program_curve.uniform2i("vp_size", width, height);
	_base_gl_program_fps.uniform2i("vp_size", width, height);
	CHECK_GL_ERROR();

	// -------- -------- -------- --------
	// resize texture

	// output
	_base_gl_texture_output.target(GLPP::Texture2DMultisample).destroy().create()
		.storage2DMultisample(_multisample_output ? _samples : 1,
			_enableSRGBCorrection ? GL_SRGB8_ALPHA8 : GL_RGBA8, width, height, GL_TRUE);
	_base_gl_framebuffer_output.texture2D(GL_COLOR_ATTACHMENT0, _base_gl_texture_output, 0);
	if (_enable_stencil_buffer) {
		_base_gl_texture_output_stencil.target(GLPP::Texture2DMultisample).destroy().create()
			.storage2DMultisample(_multisample_output ? _samples : 1,
				GL_DEPTH24_STENCIL8, width, height, GL_TRUE);
		_base_gl_framebuffer_output.texture2D(GL_DEPTH_STENCIL_ATTACHMENT,
			_base_gl_texture_output_stencil, 0);
	}
	CHECK_GL_ERROR();

	// output scale
	_base_gl_texture_output_scale.destroy().create()
		.storage2D(1, GL_RGBA8, width, height);
	_base_gl_texture_output_scale.parameterf(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	_base_gl_texture_output_scale.parameterf(GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		.parameterf(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		.parameterf(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	_base_gl_framebuffer_output_scale.texture2D(GL_COLOR_ATTACHMENT0,
		_base_gl_texture_output_scale, 0);

	// output integral sample
	_base_gl_texture_output_integrate_samples.destroy().create()
		.storage2D(1, GL_RGBA8, width, height);
	_base_gl_framebuffer_output_integrate_samples.texture2D(GL_COLOR_ATTACHMENT0,
		_base_gl_texture_output_integrate_samples, 0);

	CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::loadMask() {
	//HammersleyMaskHelper _helper;
	//_helper.loadMask();

	//_helper.getAMask(_mask.a, 32);
	//_helper.getPMask(_mask.p, 32);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::unifyCurveVertexOrder() {
	std::vector<float2> vertex;
	std::vector<uint32_t> curve_vertex_pos;
	std::vector<uint8_t> curve_type;
	std::vector<uint8_t> curve_reversed;

	// -------- -------- -------- --------
	// bind old input pack
	auto &_raw_in = _vg_in_curve[0];
	auto &_fill_in = _vg_in_curve[1];
	auto &_path_in = _vg_in_path[0];
	
	auto &_in = _gpu_stroke_to_fill ? _raw_in : _fill_in;

	_in.vertex.get(vertex);
	_in.curve_vertex_pos.get(curve_vertex_pos);
	_in.curve_type.get(curve_type);
	curve_reversed.resize(_in.n_curves);

	for (uint32_t i = 0; i < _in.n_curves; ++i) {
		auto ctype = curve_type[i];
		auto cvpos = curve_vertex_pos[i];

		auto vn = ctype & 7;

		float2 vf = vertex[cvpos];
		float2 vl = vertex[cvpos + vn - 1];

		uint32_t ixf = *((uint32_t*)(&vf.x));
		uint32_t ixl = *((uint32_t*)(&vl.x));

		uint32_t iyf = *((uint32_t*)(&vf.y));
		uint32_t iyl = *((uint32_t*)(&vl.y));

		if (iyf < iyl || (iyf == iyl && ixf < ixl)) {
			curve_reversed[i] = 0;
		} else {
			curve_reversed[i] = 1;
			for (int k = 0; k < vn / 2; ++k) {
				std::swap(vertex[cvpos + k], vertex[cvpos + vn - k - 1]);
			}
		}
	}

	_in.vertex.set(vertex);
	_in.curve_reversed.set(curve_reversed);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::clear() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::setInputTransform(const glm::mat4x4 &mat) {
	for (int i = 0; i < 4; ++i) {
		_input_transform[i].x = mat[0][i];
		_input_transform[i].y = mat[1][i];
		_input_transform[i].z = mat[2][i];
		_input_transform[i].w = mat[3][i];
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::setOutputTransform(const glm::mat4x4 &mat) {
	for(int i = 0; i < 4; ++i) {
		_output_transform[i].x = mat[0][i];
		_output_transform[i].y = mat[1][i];
		_output_transform[i].z = mat[2][i];
		_output_transform[i].w = mat[3][i];
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::setRevTransform(const glm::vec4 &rx, const glm::vec4 &ry, const glm::vec4 rw,
	const glm::vec4 rp, float a) {

	_inv_projection_context[0] = rx.x;
	_inv_projection_context[1] = rx.y;
	_inv_projection_context[2] = rx.z;
	_inv_projection_context[3] = ry.x;
	_inv_projection_context[4] = ry.y;
	_inv_projection_context[5] = ry.z;
	_inv_projection_context[6] = rw.x;
	_inv_projection_context[7] = rw.y;
	_inv_projection_context[8] = rw.z;
	_inv_projection_context[9] = rp.x;
	_inv_projection_context[10] = rp.y;
	_inv_projection_context[11] = rp.z;
	_inv_projection_context[12] = a;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

void VGRasterizer::addInk(const std::vector<float2> &vertex, 
	const std::vector<uint32_t> &color,
	bool newPath) {

	// -------- -------- -------- --------
	// bind old input pack
	auto &_raw_in = _vg_in_curve[0];
	auto &_fill_in = _vg_in_curve[1];
	auto &_path_in = _vg_in_path[0];

	//auto &_in = _gpu_stroke_to_fill ? _raw_in : _fill_in;
	auto &_in = _fill_in;

	if (newPath) {
		_currentPathId = _nextPathIndex;
		++_nextPathIndex;

		// 
		cudaMemcpy(_path_in.fill_info.gptr() + _currentPathId,
			color.data(), sizeof(uint32_t),
			cudaMemcpyHostToDevice);
	}

	auto vn = vertex.size();
	auto pn = color.size();
	assert(pn * 8 == vn);

	// TODO: check reserved size.
	//if (_usedInkNumber + pn >= _reservedInkNumber) {
	//	return;
	//}

	// gen curve info
	uint32_t new_line_number = (uint32_t)vertex.size() / 2;
	std::vector<int> curve_vertex_pos(new_line_number);
	std::vector<int> curve_path_id(new_line_number);
	for (uint32_t i = 0; i < new_line_number; ++i) {
		curve_vertex_pos[i] = _nextVertexIndex + i * 2;
		curve_path_id[i] = _currentPathId;
	}
	cudaMemcpy(_in.curve_path_id.gptr() + _nextCurveIndex, curve_path_id.data(),
		sizeof(uint32_t) * new_line_number, cudaMemcpyHostToDevice);
	cudaMemcpy(_in.curve_vertex_pos.gptr() + _nextCurveIndex, curve_vertex_pos.data(),
		sizeof(uint32_t) * new_line_number, cudaMemcpyHostToDevice);
	_nextCurveIndex += new_line_number;

	// gen vertex info
	std::vector<int> vertex_path_id(vertex.size());
	for (auto &pid : vertex_path_id) { pid = _currentPathId; }
	cudaMemcpy(_in.vertex_path_id.gptr() + _nextVertexIndex, vertex_path_id.data(), 
		vn * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(_in.vertex.gptr() + _nextVertexIndex, vertex.data(), vn * sizeof(float2),
		cudaMemcpyHostToDevice);
	_nextVertexIndex += (int)vn;

	_path_in.n_paths = _nextPathIndex;
	_in.n_curves = _nextCurveIndex;
	_in.n_vertices = _nextVertexIndex;

	//printf("u/r %d/%d\n", _usedInkNumber, _reservedInkNumber);
}

void VGRasterizer::startTextInput(int x, int y, float h_font, uint32_t color) {
	if (!_hfont) { return; }

	//if (!_text_input_contexts.empty()) {
	//	insertChar('\b');
	//}

	//_text_input_contexts.clear();
	// insert a '\n'
	TTextInputContext ctx;
	ctx.x = (float)x;
	ctx.y = (float)y;
	ctx.h_font = h_font;
	ctx.color = color;
	ctx.ch_last = '\n';

	ctx.curve_index = _nextCurveIndex;
	ctx.path_index = _nextPathIndex;
	ctx.vertex_index = _nextVertexIndex;

	_text_input_contexts.push_back(ctx);
	insertChar('|');
}

void VGRasterizer::insertChar(int ch) {
}

int VGRasterizer::isTextInputStarted() {
	if (_text_input_contexts.empty()) { return 0; }
	TTextInputContext ctx = _text_input_contexts.back();
	if (!(ctx.h_font > 0.f)) { return 0; }
	return 1;
}

void VGRasterizer::onCharMessage(int ch) {
	TTextInputContext ctx = _text_input_contexts.back();
	if (!(ctx.h_font > 0.f)) { return; }

	if (ch == '\r' || ch == ';') {
		insertChar('\b');
		ctx = _text_input_contexts.back();
		ctx.y -= ctx.h_font;
		ctx.x = _text_input_contexts[0].x;
		_text_input_contexts.push_back(ctx);
		insertChar('|');

	}
	//else if (ch == '\r' || ch == '\n') {
	else if (ch == ']') {
		insertChar('\b');

	}
	else if (ch >= ' '&&ch < 126 || ch == '\b') {
		//printf("%d\n", _in.n_curves);
		insertChar('\b');
		//printf("%d\n", _in.n_curves);
		insertChar(ch);
		//printf("%d\n", _in.n_curves);
		insertChar('|');
		//printf("%d\n", _in.n_curves);

	}
	else if (ch == 27) {
		stopTextInput();

	}
}

void VGRasterizer::stopTextInput() {
	insertChar('\b'); //remove the cursor

	//TTextInputContext ctx = _text_input_contexts.back();
	//_text_input_contexts.pop_back();
	//_text_input_contexts.clear();
	//ctx.h_font = 0.f;
	//ctx.ch_last = '\n';
	//_text_input_contexts.push_back(ctx);
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::drawOutputResize(GLuint ifbo, GLuint ofbo) {

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_STENCIL_TEST);
	glPointSize(1);
	glLineWidth(1);
	glClearColor(0, 0, 0, 1);

	glViewport(0, 0, _output_buffer_size.x, _output_buffer_size.y);

	glBlitNamedFramebuffer(ifbo, _base_gl_framebuffer_output_scale,
		0, 0, _output_buffer_size.x, _output_buffer_size.y,
		0, 0, _output_buffer_size.x, _output_buffer_size.y,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);

	glBindFramebuffer(GL_FRAMEBUFFER, ofbo);
	glClear(GL_COLOR_BUFFER_BIT);

	//_base_gl_program_output_scale.uniform2i("vp_translate", 0, 0);
	//_base_gl_program_output_scale.uniform1f("vp_scale", 1.f);

	if (_enable_output_transform) {
		_base_gl_program_output_scale.uniformMatrix4fv("o_tmat", 1, GL_TRUE,
			(float*)&(_output_transform[0]));
	}
	else {
		float tf[16] = {
			1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f
		};
		_base_gl_program_output_scale.uniformMatrix4fv("o_tmat", 1, GL_FALSE, tf);
	}

	_base_gl_texture_output_scale.bindUnit(0);
	_base_gl_vertex_array_empty.bind();
	_base_gl_program_output_scale.use();

	glDrawArrays(GL_QUADS, 0, 4);
	_base_gl_program_output_scale.disuse();
	_base_gl_vertex_array_empty.unbind();
	DEBUG_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::drawCurve() {

	// -------- -------- -------- --------
	// bind old input pack
	//auto &_raw_in = _vg_in_curve[0];
	auto &_fill_in = _vg_in_curve[1];
	auto &_path_in = _vg_in_path[0];

	auto &_in =_fill_in;

	// copy & translate all vertex.
	std::vector<float2> curve_vertex;
	std::vector<frgba> curve_color;

	_in.vertex.get(curve_vertex);

	if (!_gpu_stroke_to_fill) {

		for (auto &v : curve_vertex) {
			float4 iv4 = make_float4(v.x, v.y, 0, 1);
			float4 ov4;
			ov4.x = dot(_input_transform[0], iv4);
			ov4.y = dot(_input_transform[1], iv4);
			ov4.z = dot(_input_transform[2], iv4);
			ov4.w = dot(_input_transform[3], iv4);
			ov4 /= ov4.w;
			v.x = ov4.x;
			v.y = ov4.y;
		}

	}

	// color
	auto color_list = genHSVColorList(12);

	// tesselation
	std::vector<float2> curve_tes_vertex;

	std::function<void(float2 *, int, int)> curve_tes;
	curve_tes = [&](float2 cv[4], int curve_type, int depth) {

		float2 vfirst, vlast;
		vfirst = cv[0];
		switch (curve_type) {
		default:
			break;
		case CT_Linear:
			vlast = cv[1];
			break;
		case CT_Quadratic:
			vlast = cv[2];
			break;
		case CT_Cubic:
			vlast = cv[3];
			break;
		case CT_Rational:
			vlast = cv[2];
			break;
		}

		int ignore_cnt[4] = { 0 };
		for (int i = 0; i < (curve_type & 7); ++i) {
			if (cv[i].x < 0) { ++ignore_cnt[0]; }
			if (cv[i].y < 0) { ++ignore_cnt[1]; }
			if (cv[i].x > _output_buffer_size.x) { ++ignore_cnt[2]; }
			if (cv[i].y > _output_buffer_size.y) { ++ignore_cnt[3]; }
		}
		for (int i = 0; i < 4; ++i) {
			if (ignore_cnt[i] == (curve_type & 7)) { return; }
		}

		//
		static const float TLEN = 1;
		if ( (abs(vfirst.x - vlast.x) < TLEN && abs(vfirst.y - vlast.y) < TLEN) 
		||
			(depth > 16)
			)
		
		{
			curve_tes_vertex.push_back(vfirst);
			curve_tes_vertex.push_back(vlast);
			return;
		}

		switch (curve_type) {
		default:
			break;
		case CT_Linear:
		{
			float2 v0 = cv[0];
			float2 v1 = cv[1];
			float2 left[4];
			float2 right[4];

			left[0] = v0;
			left[1] = right[0] = (v0 + v1) *.5f;
			right[1] = v1;

			curve_tes(left, curve_type, depth+1);
			curve_tes(right, curve_type, depth + 1);
			break;
		}
		case CT_Quadratic:
		{
			float2 q0 = cv[0];
			float2 q1 = cv[1];
			float2 q2 = cv[2];

			float2 l_left = (q0 + q1) * 0.5f;
			float2 l_right = (q1 + q2) * 0.5f;

			float2 p = (l_left + l_right) * 0.5f;

			float2 left[4];
			float2 right[4];

			left[0] = q0;
			left[1] = l_left;
			left[2] = p;

			right[0] = p;
			right[1] = l_right;
			right[2] = q2;

			curve_tes(left, curve_type, depth + 1);
			curve_tes(right, curve_type, depth + 1);
			break;
		}
		case CT_Cubic:
		{
			float2 c0 = cv[0];
			float2 c1 = cv[1];
			float2 c2 = cv[2];
			float2 c3 = cv[3];

			float2 q0 = (c0 + c1) * .5f;
			float2 q1 = (c1 + c2) * .5f;
			float2 q2 = (c2 + c3) * .5f;

			float2 l0 = (q0 + q1) * .5f;
			float2 l1 = (q1 + q2) * .5f;

			float2 p = (l0 + l1) * .5f;

			float2 left[4];
			float2 right[4];

			left[0] = c0;
			left[1] = q0;
			left[2] = l0;
			left[3] = p;

			right[0] = p;
			right[1] = l1;
			right[2] = q2;
			right[3] = c3;

			curve_tes(left, CT_Cubic, depth + 1);
			curve_tes(right, CT_Cubic, depth + 1);
			break;
		}
		case CT_Rational:
		{
			float2 out[4];

			auto blossom = [](float2 *B, float Bw, float u, float v, float &w) -> float2
			{
				float uv = u*v;
				float b0 = uv - u - v + 1,
					b1 = u + v - 2 * uv,
					b2 = uv;

				w = 1 * b0 + Bw*b1 + 1 * b2;

				return B[0] * b0 + B[1] * b1 + B[2] * b2;
			};

			auto subcurve = [&](float u, float v) {

				float2 cB[3] = { cv[0], cv[1], cv[2] };
				float cBw = cv[3].x;

				float wA, wB, wC;
				float2 A = blossom(cB, cBw, u, u, wA);
				float2 B = blossom(cB, cBw, u, v, wB);
				float2 C = blossom(cB, cBw, v, v, wC);

				float s = 1.0f / sqrt(wA * wC);
				out[1] = s*B;
				//out.w = s*wB;
				out[3].x = s*wB;

				if (u == 0)
				{
					out[0] = cB[0];
					out[2] = C / wC;
				}
				else if (v == 1)
				{
					out[0] = A / wA;
					out[2] = cB[2];
				}
				else
				{
					out[0] = A / wA;
					out[2] = C / wC;
				}
				//return out;
			};

			subcurve(0.f, 0.5f);
			curve_tes(out, curve_type, depth + 1);

			subcurve(0.5f, 1.f);
			curve_tes(out, curve_type, depth + 1);

			break;
		}

		}
	};

	std::vector<uint32_t> cpu_curve_vertex_pos;
	std::vector<uint8_t> cpu_curve_type;
	std::vector<float> cpu_w;
	std::vector<uint32_t> cpu_path_id;
	std::vector<uint64_t> cpu_is_path_visible;

	_in.curve_vertex_pos.get(cpu_curve_vertex_pos);
	_in.curve_type.get(cpu_curve_type);
	_in.curve_arc_w.get(cpu_w);
	_in.curve_path_id.get(cpu_path_id);
	_base_gpu_is_path_visible.get(cpu_is_path_visible);

	if (_enable_output_transform) {
		for (auto &v : curve_vertex) {
			float4 iv4 = make_float4(v.x, v.y, 0, 1);
			float4 ov4;
			ov4.x = dot(_output_transform[0], iv4);
			ov4.y = dot(_output_transform[1], iv4);
			ov4.z = dot(_output_transform[2], iv4);
			ov4.w = dot(_output_transform[3], iv4);
			ov4 /= ov4.w;
			v.x = ov4.x;
			v.y = ov4.y;
		}
	}

	for (uint32_t i = 0; i < _in.n_curves; ++i) {

		auto curve_type = cpu_curve_type[i];
		auto curve_v0_pos = cpu_curve_vertex_pos[i];

		float2 cv[4];

		for (int k = 0; k < (curve_type & 7); ++k) {
			cv[k] = ((float2*)curve_vertex.data() + curve_v0_pos)[k];
		}
		if (curve_type == CT_Rational) {
			float v = cpu_w[i];
			cv[3].x = v;
			cv[1] *= v;
		}
		curve_tes(cv, curve_type, 0);

		frgba color;

		auto curve_path_id = cpu_path_id[i];
		auto path_visible = cpu_is_path_visible[curve_path_id];

		if (PATH_VISIBLE(path_visible)) {
			int rindex = rand() % 12;
			color = color_list[rindex];
		}
		else {
			color = frgba(0.25, 0.25, 0.25, 0.5);
		}

		while (curve_color.size() < curve_tes_vertex.size()) {
			curve_color.push_back(color);
		}
	}

	NamedBuffer _gl_buffer_curve_vertex;
	NamedBuffer _gl_buffer_curve_color;

	_gl_buffer_curve_vertex.create().data((sizeof(float2) * (GLsizei)curve_tes_vertex.size()),
		curve_tes_vertex.data(), GL_STATIC_DRAW);

	_gl_buffer_curve_color.create().data(sizeof(float4) * (GLsizei)curve_color.size(),
		curve_color.data(), GL_STATIC_DRAW);

	NamedTexture _gl_texture_curve_vertex;
	NamedTexture _gl_texture_curve_color;

	_gl_texture_curve_vertex.target(GLPP::TextureBuffer).create().buffer(
		GL_RG32F, _gl_buffer_curve_vertex).bindUnit(0);
	_gl_texture_curve_color.target(GLPP::TextureBuffer).create().buffer(
		GL_RGBA32F, _gl_buffer_curve_color).bindUnit(1);

	// draw
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_STENCIL_TEST);
	glDisable(GL_BLEND);
	glPointSize(1.f);
	glLineWidth(1.f);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, _output_buffer_size.x, _output_buffer_size.y);

	_base_gl_program_curve.use();
	glDrawArrays(GL_LINES, 0, (GLsizei)curve_tes_vertex.size());
	_base_gl_program_curve.disuse();

	_gl_texture_curve_color.destroy();
	_gl_texture_curve_vertex.destroy();

	_gl_buffer_curve_color.destroy();
	_gl_buffer_curve_vertex.destroy();

}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::rasterize() {

	// count fps & set to window title.
	static std::list<int> timestamps;
	int now = clock();
	while (!timestamps.empty() && timestamps.front() + 1000 < now) {
		timestamps.pop_front();
	}
	timestamps.push_back(now);

	float fps = 1000.0f / ((timestamps.back() - timestamps.front()) / (float)timestamps.size());
	_base_gl_program_fps.uniform1i("fps", (GLint)fps);

	rasterizeImpl();

	glFinish();

	if (!_multisample_output) {
		glBlitNamedFramebuffer(
			_base_gl_framebuffer_output,
			_base_gl_framebuffer_output_integrate_samples,
			0, 0, _output_buffer_size.x, _output_buffer_size.y,
			0, 0, _output_buffer_size.x, _output_buffer_size.y,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);
	}
	else {
		glBlitNamedFramebuffer(
			_base_gl_framebuffer_output,
			_base_gl_framebuffer_output_integrate_samples,
			0, 0, _output_buffer_size.x, _output_buffer_size.y,
			0, 0, _output_buffer_size.x, _output_buffer_size.y,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);
	}

	glFinish();

	if (_enable_output_transform) {
		drawOutputResize(_base_gl_framebuffer_output_integrate_samples, 0);
	}
	else {
		glBlitNamedFramebuffer(_base_gl_framebuffer_output_integrate_samples, 0,
			0, 0, _vp_size.x, _vp_size.y,
			_vp_pos.x, _vp_pos.y,
			_vp_pos.x + _vp_size.x,
			_vp_pos.y + _vp_size.y,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);
	}

	if (_save_output_to_file) { 
		saveOutputToFile();
	}

	if (_draw_curve) {
		drawCurve();
	}

	glFinish();

	if (_enable_step_timing) {
		long long ts;
		QueryPerformanceCounter((LARGE_INTEGER*)&ts);
		_step_timestamp.push_back(ts);
	}

}

void VGRasterizer::rasterize(bool verbose) {
	auto old_verbose = _verbose;
	_verbose = verbose;
	rasterize();
	_verbose = old_verbose;
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::vector<double> VGRasterizer::stepTiming() {
	LARGE_INTEGER pc_frequency;
	QueryPerformanceFrequency(&pc_frequency);

	_step_timing.clear();

	for (int i = 0; i+1 < _step_timestamp.size(); ++i) {
		double diff = (double)(_step_timestamp[i + 1] - _step_timestamp[i]);
		diff /= *(long long*)&pc_frequency;
		_step_timing.push_back(diff);
	}

	return _step_timing;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::saveOutputToFile() {

	auto w = _output_buffer_size.x;
	auto h = _output_buffer_size.y;

	NamedTexture save_texture;
	NamedFramebuffer save_framebuffer;

	save_texture.target(GLPP::Texture2D).create().storage2D(1, 
		_enableSRGBCorrection ? GL_SRGB8_ALPHA8 : GL_RGBA8, w, h);
	save_framebuffer.create().texture2D(GL_COLOR_ATTACHMENT0, save_texture, 0);

	glBlitNamedFramebuffer(
		_base_gl_framebuffer_output,
		save_framebuffer,
		0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, save_framebuffer);

	int size = w * h * 4;
	uint8_t *data = new uint8_t[size];

	glReadnPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, size, data);
	glFinish();

	int stride = w * 4;
	uint8_t *swap = new uint8_t[stride];

	if (!_vertical_flip_output_file) {
		for (int i = 0; i < (h / 2); ++i) {
			auto p0 = data + stride * i;
			auto p1 = data + stride * (h - i - 1);

			memcpy(swap, p0, stride);
			memcpy(p0, p1, stride);
			memcpy(p1, swap, stride);
		}
	}

	for (int i = 0; i < w*h*4; i += 4) {
		data[i+3] = 255;
	}

	if (_enableSRGBCorrection) {
		for (int i = 0; i < w*h * 4; i += 4) {
			data[i + 3] = 255;
		}
	}

	stbi_write_png(_output_file_name.c_str(), w, h, 4, data, stride);
	delete[] data;

	save_framebuffer.destroy();
	save_texture.destroy();

	exit(0);
}

std::string VGRasterizer::tigerClipString() {
	return 
		"M429.3603516,419.947998c-2.8027344-7.3859863-7.800415-21.6293335-8.0012207-26.0792847"
		"c0.8155518,0.3980103,2.2023315,1.2397461,3.0715332,2.3844604c0.8115234,1.2885742,2.3457031,3.4174805,3.4277344,3.230957"
		"l0.1425781-0.0249023l0.1044922-0.0996094c1.1757813-1.1171875,0.9736328-2.8295898-1.2773438-10.7963867"
		"c-3.0673828-9.5996094-4.7802734-18.175293-3.7412109-18.7260742c0-0.003418,0.5673828-0.199707,2.5322266,2.3007813"
		"c0.3554688,0.4262695,2.1953125,2.5258789,3.5068359,1.871582c0.8320313-0.4150391,1.5849609-1.7475586,0.0136719-8.4262695"
		"c-2.9287109-12.4477539-5.0878906-18.9526367-5.8544922-21.1074219c0.5292969,0.0385742,1.1640625-0.0737305,1.6201172-0.6342773"
		"c2.2070313-2.706543-3.0625-13.5849609-15.6660156-32.3457031l8.1503906,3.3339844l-0.5625-1.1367188"
		"c-0.7705078-1.5561523-19.0791016-38.1191406-39.7324219-43.1938477l-7.15625-5.3666992"
		"c0.5440063-0.5653076,1.628479-1.7197876,3.0595703-3.3743286c2.1447144-1.6366577,4.8807983-4.3008423,7.3399658-8.4622192"
		"c4.9869995-8.4390259,19.5640259-26.4689941,16.1119995-50.6360474c0,0,1.1500244-10.7409668-4.6040039-11.1239624"
		"c0,0-8.0549927-1.5350342-14.960022,5.7539673c0,0-6.5219727,3.0679932-8.822998,2.6849976l-1.194397,0.0407104"
		"c-0.2001953-0.0380859-0.4008179-0.067749-0.6000366-0.1209106l-0.0703125-0.0185547l-0.0722656,0.0029297"
		"c-0.2949219,0.0112305-29.5498047,1.1503906-31.375,1.1572266c-0.2314453-0.0708008-1.1533203-0.7841797-2.1298828-1.5395508"
		"c-0.8892212-0.6878052-1.7842407-1.3578491-2.6838379-2.0133057c-2.114563-1.6812134-5.1712646-3.8253784-9.3128052-6.0875244"
		"c-24.6345215-14.3878174-52.7958374-17.50354-83.9522705-9.222168c-2.0392456,0.4074707-3.9735107,0.8063965-5.6149902,1.1660156"
		"c-1.1587524,0.0648193-2.3861694,0.0071411-3.5513306-0.295166c-1.5496826-1.2754517-19.4625854-14.645752-64.7297974,6.4546509"
		"c-2.0930176,0.197876-7.7946777,0.812439-12.2161865,1.9921875c-1.7927856,0.1986694-4.7960205,1.192627-11.0432739,6.1908569"
		"c-5.1230469,4.0986328-7.0117188,5.8149414-8.2612305,6.9506836c-0.0185547,0.0169067-0.0341187,0.0309448-0.0524292,0.0475464"
		"c-2.6581421,1.977478-4.7928467,3.4949951-5.388916,3.7333984c-1.4333496,0.5731812-7.9000854,4.7879028-11.3974609,7.0419922"
		"c-2.0770874,1.0934448-3.8753662,1.8903198-4.9839478,2.0750122l-0.152832,0.0532227"
		"c-0.4570313,0.2519531-11.1694336,6.2451172-15.4648438,15.7597656l-3.5102539,1.1699219l-0.059082,0.2651367"
		"c-0.0146484,0.0664063-1.4296875,6.4316406-1.8623047,7.8735352c-0.7607422,0.605957-4.6005859,3.8818359-5.4003906,8.8330078"
		"c-1.2451172,0.8745117-8.3574219,6.0605469-8.0976563,10.1962891c-0.2050781,0.6796875-1.5390625,5.1694336-2.2558594,9.3618164"
		"c-1.2241211,0.8354492-6.6831055,4.7348633-6.2060547,7.5004883c-0.8344727,1.5634766-6.8374023,13.0771484-6.1831055,19.5175781"
		"c-1.6386719-0.0366211-6.253418,0.0566406-8.5820313,2.0546875l-0.1323242,0.1132813l-0.0288086,0.1713867"
		"c-0.2890625,1.7314453-1.1074219,4.3911133-2.0180664,4.625c-0.043457,0.0185547-1.065918,0.4692383-1.4575195,1.6796875"
		"c-0.3398438,1.0517578-0.1005859,2.3129883,0.7109375,3.7524414c-0.3032227,0.5477905-1.034729,1.9127808-1.5037231,3.1157837"
		"c-0.6907349-0.0663452-1.3834229-0.1322632-2.1144409-0.1865845c-0.6162109-0.0571289-0.9360352-0.0664063-0.9516602-0.0664063"
		"l-0.0039063,0.0957031c0.3208008,0.0209961,0.6391602,0.0429688,0.9526367,0.0664063"
		"c0.4942627,0.0458374,1.1851196,0.1242065,2.057251,0.2523804c-0.1082764,0.2896729-0.2045898,0.5709839-0.2672119,0.8213501"
		"l-0.0273438,0.1240234l0.0283813,0.1115723c-1.4389648-0.0644531-2.9424438-0.1081543-4.5610962-0.1081543"
		"c-0.4121094,0-0.8300781,0.0019531-1.2543945,0.0068359c-0.6889648-0.0058594-1.0527344,0.0161133-1.0703125,0.0170898"
		"l0.0039063,0.0957031c0.3618164-0.0073242,0.7182617-0.0131836,1.0717773-0.0170898"
		"c1.1029053,0.0079346,3.0643311,0.0834351,5.8443604,0.4015503c0.0028687,0.2088013-0.0153809,0.5028687-0.0810547,0.8817749"
		"c-2.3203735,0.0217896-4.8665771,0.1275635-7.6915283,0.3704834c-0.6899414,0.0463867-1.0444336,0.09375-1.0620117,0.0961914"
		"l0.0112305,0.0952148c0.3579102-0.0341797,0.7133789-0.0664063,1.0634766-0.0961914"
		"c1.3114014-0.0887451,3.8477173-0.1715698,7.567688,0.0519409c-0.0668335,0.265686-0.15802,0.5617676-0.2758179,0.8865356"
		"c-2.7538452,0.2084961-5.8629761,0.5723877-9.3826904,1.1845703c-0.6816406,0.105957-1.0292969,0.1831055-1.0463867,0.1870117"
		"l0.0195313,0.0942383c0.3535156-0.0649414,0.7026367-0.1274414,1.0488281-0.1875"
		"c1.4694824-0.2280884,4.5057373-0.5853882,9.1010132-0.6289063c-0.4951782,1.1447144-1.3110352,2.5769043-2.671814,4.309082"
		"l-0.0771484,0.1420898c-0.2514648,0.7392578-6.1166992,18.0927734-4.2392578,23.2714844"
		"c0.0292969,0.5268555,0.1577148,4.2314453-1.9648438,5.5600586c-0.2695313,0.0253906-0.6357422,0.159668-0.8676758,0.5507813"
		"c-0.2224731,0.375061-0.2597656,0.9170532-0.1218262,1.6190796c-0.0056152,0.4119263,0.1200562,0.8479004,0.3892822,1.2984009"
		"c0.744873,1.9591675,2.3763428,4.7329102,4.9244385,8.3706055c0.1123047,0.2456055-0.050293,1.2172852-2.0927734,2.7705078"
		"c-1.4501953,0.34375-16.7416992,4.3295898-19.0424805,19.4091797c-0.1917725,0.210144-0.6138916,0.6762085-1.1869507,1.3249512"
		"c-1.8804321-25.850769-8.050293-36.8687744-13.1265259-41.5456543c-3.7739258-3.4770508-6.9204102-3.3764648-6.9526367-3.3769531"
		"l-0.015625,0.1625977c14.6629028,3.562561,17.6204834,32.8346558,18.5303345,46.5609741"
		"c-0.9812622,1.1477051-2.1239624,2.5200806-3.2875366,3.9927979C3.4360352,367.0695801-6.5294189,358.3554077-13.0996094,355.598877"
		"c-4.7319336-1.9858398-7.6577148-0.8217773-7.6875-0.8100586l0.0405273,0.1582031"
		"c0.4472656-0.0488281,0.8901367-0.0727539,1.3330078-0.0727539c15.0986938,0,27.7122192,27.7712402,32.6439819,39.1852417"
		"c-1.6077881,2.1134033-3.1522217,4.3280029-4.2523804,6.3058472"
		"c-18.0601196-23.0712891-30.4126587-29.1046143-37.6611328-30.0467529c-5.0908203-0.6611328-7.6030273,1.2358398-7.628418,1.2553711"
		"l0.0810547,0.1416016c15.3530273-6.0766602,37.0395508,21.0908203,44.1655273,30.0180664"
		"c0.0783691,0.0982056,0.1397095,0.1747437,0.2129517,0.2664185c-0.4867554,1.0869751-0.772522,2.0522461-0.772522,2.8209839"
		"c0,0.3616333,0.0094604,0.7479248,0.0235596,1.1464844c-21.5117188-17.3222656-34.3532104-20.3652344-41.4381104-19.7392578"
		"c-5.1118164,0.4516602-7.15625,2.8461914-7.1762695,2.8701172l0.1098633,0.1210938"
		"c13.2185059-8.9354858,38.9178467,11.177002,48.6444702,19.0172729c0.0734253,0.876709,0.1723633,1.824646,0.296936,2.8461914"
		"c-23.5493164-15.7516479-36.8529053-17.6466064-43.9291992-16.3497925"
		"c-5.0478516,0.9262695-6.8598633,3.5004883-6.8774414,3.5263672l0.1201172,0.1103516"
		"c11.6835938-9.5925903,36.9259644,5.715332,48.411377,13.2241821c-13.024231-4.2005005-21.0222168-3.8261719-25.8078613-2.2935181"
		"c-4.8881836,1.5654297-6.355957,4.3510742-6.3701172,4.3789063l0.1333008,0.0942383"
		"c7.1118774-7.5596924,22.0686646-4.295166,34.7444458,0.4265747c0.0154419,0.1037598,0.0287476,0.2008667,0.0446167,0.3058472"
		"c0.0007935,0.1521606,0.0308838,0.5543213,0.2391357,1.1020508c0.0600586,0.6865234,0.1283569,1.3966064,0.2058105,2.1228027"
		"c-13.0292358-6.3381348-21.3380127-7.0517578-26.4302979-6.1188965c-5.0478516,0.9262695-6.8598633,3.5004883-6.8774414,3.5263672"
		"l0.1201172,0.1103516c7.7984009-6.4058228,21.644043-1.7051392,33.4066162,4.3595581"
		"c0.0744629,0.5877686,0.154541,1.177002,0.2422485,1.7642212c-20.885437-6.3786011-31.9921265-5.0183105-37.7890015-2.4055176"
		"c-4.6796875,2.1079102-5.8227539,5.0415039-5.8334961,5.0708008l0.1430664,0.0791016"
		"c7.7514648-10.3798828,29.552124-5.3907471,43.7817993-0.894165c0.2947998,1.6447754,0.6517334,3.2087402,1.0806274,4.5647583"
		"c-24.4824219-4.6062012-36.3884888-1.5964966-42.1265869,2.0012817c-4.3481445,2.7265625-5.0805664,5.7880859-5.0874023,5.8183594"
		"l0.152832,0.0585938c6.944458-12.5753784,33.5866089-8.7788086,47.756897-6.0280762"
		"c0.0889282,0.1957397,0.1789551,0.3875732,0.2728882,0.5651855c-0.0608521,0.2473755-0.1679688,0.7235718-0.2776489,1.3916626"
		"c-7.9727783,0.6799316-12.9602661,2.6297607-16.0661011,4.7050171c-4.2670898,2.8505859-4.9106445,5.9326172-4.9165039,5.9638672"
		"l0.1542969,0.0537109c3.256958-6.3241577,11.4613647-8.7709961,20.6419067-9.350708"
		"c-0.2086182,1.8903198-0.3092651,4.5771484,0.119812,7.7409058c-11.9210815,2.0237427-18.2918701,5.5991211-21.6923828,8.8510132"
		"c-3.7089844,3.5478516-3.809082,6.6943359-3.8095703,6.7255859l0.1611328,0.0263672"
		"c2.6397095-8.4873047,13.9310913-12.3554688,25.5764771-14.1124878c0.4731445,2.59198,1.3162231,5.4460449,2.7322388,8.40802"
		"c-10.4502563,3.62677-15.8300171,7.8484497-18.5836182,11.387085c-3.1523438,4.0507813-2.7919922,7.1787109-2.7880859,7.2099609"
		"l0.1635742,0.0019531c1.317688-8.4271851,11.1404419-13.8052368,21.8859863-17.2630005"
		"c0.5082397,0.9544678,1.0718994,1.916626,1.7102051,2.8831177c-0.0493164,1.0888672-0.2509766,7.1347656,1.2045898,10.0449219"
		"c0.1611328,0.3378906,3.9985352,8.2919922,8.793457,9.0898438c1.3212891,0.2207031,3.2246094,0.7353516,5.6347656,1.3857422"
		"c4.1264648,1.1152344,9.7661133,2.6386719,15.7875977,3.5078125c0.0343628,0.0291748,0.0828857,0.0708618,0.1334229,0.1144409"
		"c0.5717163,1.4471436,3.0911865,7.7940063,4.774292,11.4978638c0.9093628,2.0004272,4.5631714,6.0701904,8.5410156,10.0966187"
		"c0.0401611,1.1707153-0.0600586,2.3410034-0.3491211,3.4971313l-0.0141602,0.1044922"
		"c-0.1269531,5.1904297-0.9711914,13.3193359-2.7119141,16.0185547c-0.3540039,0.3173828-0.6850586,0.6279297-0.9775391,0.9111328"
		"l0.527832,0.7832031c0.402832-0.1796875,0.7666016-0.5175781,1.0952148-0.9785156"
		"c1.7280273-1.5302734,3.3603516-2.6269531,3.762207-2.5009766c0.0620117,0.0927734,0.4892578,1.1035156-3.5507813,8.1738281"
		"l-4.6049805,19.5849609l1.1132813-0.9541016c4.0634766-3.484375,10.7929688-8.8330078,13.2978516-9.8798828"
		"c-0.3085938,1.2470703-0.8510742,1.96875-4.3852539,6.1367188l-9.6723633,25.0732422l0.7753906,0.5224609"
		"c3.6123047-3.3789063,9.1020508-8.3857422,11.7553711-10.5429688c-0.1176758,0.2167969-0.28125,0.484375-0.5380859,0.8564453"
		"l-3.7036133,9.3701172l1.2373047-0.7783203c8.0195313-5.0498047,20.0625-11.7314453,21.7265625-10.8027344"
		"c0.0258789,0.0146484,0.0429688,0.0244141,0.0512695,0.0878906c0.074707,0.5888672-0.7954102,2.9628906-10.4731445,12.2646484"
		"l0.5288086,0.7832031c0.0742188-0.0332031,7.4868164-3.3125,11.4375-0.8017578l0.1572266,0.1005859l0.1831055-0.0322266"
		"c2.175293-0.3828125,4.3500977-0.5087891,5.0776367-0.3173828c-2.7910156,1.5068359-20.0849609,11.4384766-23.2773438,27.7568359"
		"l-0.355957,1.8212891L70.75,595.3991699c1.9267578-2.2939453,4.144043-4.1347656,4.6616211-3.8564453"
		"c0.0063477,0.0039063,0.6225586,0.3857422-0.5014648,4.1787109l-0.0219727,0.0751953l0.3857422,11.203125l0.9462891,0.0908203"
		"c0.4614258-2.015625,1.1464844-4.2861328,1.6152344-5.2050781c1.046875,1.9648438,1.0439453,6.0634766,0.8911133,20.4365234"
		"l-0.0117188,1.1230469l0.8193359-0.7685547c4.0708008-3.8154297,9.0224609-7.4296875,9.9116211-6.7421875"
		"c0.1816406,0.1425781,0.8730469,1.2353516-3.3706055,8.8447266l-0.0605469,0.1083984v17.4589844l0.8310547-0.8964844"
		"c3.5615234-3.8427734,7.9204102-7.8701172,9.0175781-8.0009766c-0.0595703,0.6757813-0.8310547,2.6484375-1.1376953,3.4316406"
		"l-0.8754883,2.3046875l1.565918-1.2919922c0.8881836-0.7441406,3.2495117-2.7207031,4.7773438-2.0761719"
		"c0.7070313,0.2958984,1.6508789,1.7128906,0.0195313,7.9267578c-0.1289063,0.9433594-0.4516602,4.0703125,0.8188477,4.7041016"
		"l0.2050781,0.1025391l0.2084961-0.0947266c1.1352539-0.5185547,1.5756836-1.2529297,3.1196289-5.1914063"
		"c4.2050781-8.0410156,8.6308594-14.5078125,9.6689453-14.1269531c0.0458984,0.0166016,1.1030273,0.5195313-0.9829102,9.6650391"
		"c-0.1855469,4.4462891-0.0854492,9.8046875,1.4423828,10.0791016c0.7167969,0.1308594,1.46875-0.5146484,2.6660156-3.7626953"
		"c0.6240234,3.3496094,2.6469727,10.1142578,9.1181641,15.1787109l0.7988281,0.625l-0.0239258-1.0146484"
		"c-0.230957-9.7900391,0.4819336-25.5224609,3.3798828-26.0576172c0.7026367-0.1347656,2.8095703,0.7753906,6.7998047,11.9902344"
		"l4.2509766,17.4287109l0.5092773-1.6748047c0.109375-0.3583984,2.6362305-8.7333984,2.3349609-13.7773438"
		"c2.1489258-2.3398438,5.9702148-5.5849609,7.515625-4.7275391c0.8339844,0.4580078,1.8388672,2.640625-0.796875,12.0986328"
		"l0.8608398,0.3945313c4.953125-7.4287109,9.9428711-13.3427734,10.909668-12.9326172"
		"c0.0048828,0.0029297,0.4594727,0.3505859-0.2338867,3.2539063c-0.3012695,0.6337891-7.3540039,15.5742188-5.7758789,20.3056641"
		"l0.3735352,1.1201172l0.5131836-1.0634766c3.8632813-8.0009766,13.1953125-27.2216797,16.2700195-33.1240234"
		"c-0.2036133,6.0166016-0.2670898,18.0458984,2.4418945,18.5087891c1.2558594,0.2207031,3.0175781-1.2001953,6.6381836-12.9785156"
		"c1.2558594,3.0019531,4.2280273,10.8994141,2.2006836,14.0849609l0.7436523,0.5966797"
		"c0.7885742-0.7880859,7.5546875-7.6669922,7.0703125-10.9921875c0.440918-0.7138672,1.5483398-2.1210938,2.6337891-1.9443359"
		"c0.7792969,0.1220703,2.269043,1.2109375,3.4868164,7.125c0.4814453,2.5166016,1.5039063,6.7666016,2.9345703,6.9648438"
		"c0.1699219,0.0224609,0.3349609,0.0009766,0.4892578-0.0683594c0.8398438,4.7861328,2.0957031,10.0185547,3.5722656,10.0185547"
		"c0.0068359,0,0.0146484,0,0.0224609,0c0.8037109-0.0292969,1.7011719-0.8193359,2.1767578-9.8525391"
		"c0.0263672-0.2236328,0.6289063-5.5556641-0.7587891-12.3701172c-0.9033203-4.4384766-2.4453125-8.4560547-4.5820313-11.9443359"
		"c0.0664063-0.4013672,0.1640625-1.3457031-0.0791016-2.6337891c1.7285156,2.1865234,3.8007813,4.2226563,5.1464844,3.6523438"
		"c1.1474609-0.4902344,2.0927734-2.5068359-0.5849609-13.6181641c2.9765625,2.0625,9.9189453,6.8105469,11.0341797,7.0244141"
		"l0.8710938,0.1669922l-0.3378906-0.8203125c-0.1171875-0.2851563-0.5146484-1.0136719-1.1152344-2.1162109"
		"c-5.0664063-9.2949219-7.1552734-14.4863281-6.2109375-15.4316406c0.0917969-0.0927734,0.5126953-0.3310547,2.0488281,0.5058594"
		"l1.3417969,0.7324219l-0.6835938-1.3681641c-0.3828125-0.765625-0.9697266-2.5273438-0.3388672-3.0830078"
		"c0.5273438-0.46875,2.7441406-0.8876953,11.9208984,5.2304688l0.6181641-0.7246094"
		"c-6.0224609-6.5429688-5.2822266-7.5976563-5.2480469-7.6396484c0.2626953-0.3173828,2.1826172-0.0322266,5.6230469,2.0957031"
		"l0.484375,0.3076172c1.9238281,1.2226563,2.984375,1.8955078,4.0136719,0.9150391l0.1816406-0.1728516l-0.0380859-0.2470703"
		"c-0.1875-1.2314453-0.8808594-2.3164063-3.0986328-4.8486328l-0.5742188-0.6210938"
		"c-2.9296875-3.1689453-3.7470703-4.1015625-3.7109375-5.4326172c2.1015625,0.7207031,4.6621094,2.9814453,10.859375,8.6162109"
		"c3.7119141,5.25,10.7177734,15.4589844,11.7119141,17.9912109c0.0302734,0.0878906,0.0595703,0.1738281,0.0878906,0.2578125"
		"l0.9150391-0.2871094c-0.0205078-0.0732422-0.0488281-0.15625-0.0839844-0.2470703"
		"c-0.9609375-2.7851563-9.3828125-26.9462891-13.6425781-30.5009766c0.5527344-2.0371094,4.1855469-13.9033203,15.0712891-19.7226563"
		"c8.3984375-4.4892578,19.0244141-4.3759766,31.5859375,0.3417969c0.5576172,1.3525391,3.2197266,7.3710938,6.4082031,7.0996094"
		"c2.1113281-0.1494141,3.7832031-2.8583984,5.1083984-8.2783203c0.7080078-0.2714844,3.2578125-1.0488281,6.6757813,0.2529297"
		"c3.9970703,1.5224609,9.9277344,6.2138672,15.4863281,19.9121094l0.5166016,1.2724609l0.3876953-1.3173828"
		"c0.3681641-1.2529297,3.3769531-11.6191406,3.1757813-15.2802734c4.4375,0.7304688,5.3056641,0.3056641,5.5507813,0.0429688"
		"c3.7783203,1.2480469,11.2089844,3.5771484,13.2929688,3.3486328c1.1201172,1.0976563,4.8623047,4.5703125,6.6230469,3.9541016"
		"c0.2744141-0.0966797,0.6015625-0.3144531,0.7861328-0.8046875c1.8164063,0.5185547,5.8105469,1.4824219,6.96875,0.3203125"
		"c1.0605469,1.8759766,2.9609375,5.3046875,4.6425781,8.6914063c1.6933594,3.4091797,3.0117188,6.1708984,3.2910156,8.0996094"
		"l0.9472656,0.0126953l2.1074219-12.2919922l1.9023438,2.6621094l0.2080078-1.1035156"
		"c0.1191406-0.6289063,1.0283203-5.5263672,0.890625-7.8359375c3.7978516,1.2871094,18.4521484,7.3808594,22.7226563,25.8886719"
		"l2.265625,9.5136719l0.5634766-1.390625c0.2236328-0.5507813,5.1552734-12.7958984,4.4326172-17.5654297"
		"c1.3193359,0.3769531,3.9726563,1.4902344,4.2353516,4.3867188l0.9501953,0.0380859"
		"c0.1523438-0.8837891,3.5566406-20.9824219-0.3857422-27.6835938c1.1582031-0.0400391,3.3251953,0.2246094,4.0625,2.4355469"
		"l0.9345703-0.1513672v-6.3769531c1.4208984,0.1132813,5.0478516,0.28125,6.3662109-0.8984375"
		"c0.2880859-0.2587891,0.4648438-0.5751953,0.5195313-0.9257813c0.4638672-0.3769531,2.0048828-1.5205078,3.3691406-1.2060547"
		"c0.7792969,0.1748047,1.4072266,0.8203125,1.8652344,1.9189453l0.8945313-0.34375"
		"c-2.4492188-6.9541016-5.4267578-18.5449219-2.9892578-20.0439453c0.6962891-0.4296875,2.65625-0.3271484,7.8916016,5.125"
		"c0.5878906,0.8828125,2.625,3.7050781,4.1523438,3.0849609c0.9921875-0.40625,1.9082031-2.0371094,0.1708984-11.1123047"
		"c-1.4355469-7.4951172-2.9619141-12.2294922-3.8740234-15.0585938c-0.6728516-2.0869141-1.1171875-3.4648438-0.8349609-3.8896484"
		"c0.1865234-0.2792969,0.9101563-0.375,1.5927734-0.4326172l0.3496094-0.0292969l0.078125-0.3417969"
		"c0.0712891-0.3144531,0.5888672-2.7441406-0.1796875-4.3486328c0.3857422,0.1523438,0.9130859,0.2539063,1.390625-0.1142578"
		"c0.9541016-0.7314453,1.8310547-3.3466797-1.4697266-17.4648438c0.2509766-0.078125,0.5205078-0.2324219,0.7607422-0.5097656"
		"c2.0439453-2.3642578-0.0888672-11.4541016-6.1689453-26.296875c0.5878906-0.7900391,1.8378906-3.3935547-0.7021484-9.3867188"
		"c2.3720703,1.1728516,7.34375,3.3271484,9.5917969,2.0410156l0.3154297-0.1796875l-0.0888672-0.3525391"
		"c-0.0166016-0.0664063-0.4404297-1.6640625-3.4970703-5.4970703c-5.0283203-12.7705078-11.8339844-33.6728516-8.7607422-35.6064453"
		"c0.4707031-0.2978516,2.0703125-0.421875,6.828125,4.7089844c0.7324219,0.8378906,4.4921875,4.9599609,6.6591797,3.7880859"
		"C431.1269531,432.4733887,432.3076172,430.4411621,429.3603516,419.947998z";
}

} // end of namespace RasterizerBase

} // end of namespace Mochimazui

