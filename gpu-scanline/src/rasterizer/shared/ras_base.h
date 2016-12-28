
#ifndef _MOCHIMAZUI_VG_RASTERIZER_BASE_H_
#define _MOCHIMAZUI_VG_RASTERIZER_BASE_H_

#include <map>

#include <cuda.h>
#include <host_defines.h>

#include <mochimazui/3rd/gl_4_5_compatibility.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>

#include <mochimazui/glpp.h>
#include <mochimazui/cuda_array.h>

#include "ras_pipeline_mode.h"

namespace Mochimazui {

class VGContainer;

namespace RasterizerBase {

using GLPP::NamedBuffer;
using GLPP::NamedFramebuffer;
using GLPP::NamedTexture;
using GLPP::ShaderProgram;

using CUDATL::CUDAArray;

// -------- -------- -------- -------- -------- -------- -------- --------
struct VGInputCurveDataPack {

	// vertex data
	CUDAArray<float2> vertex;
	CUDAArray<uint32_t> vertex_path_id;

	// curve
	CUDAArray<uint32_t> curve_vertex_pos;
	CUDAArray<uint8_t> curve_type;
	CUDAArray<uint32_t> curve_path_id;
	CUDAArray<uint32_t> curve_contour_id;
	CUDAArray<float> curve_arc_w;
	CUDAArray<float> curve_offset;
	CUDAArray<uint8_t> curve_reversed;

	// numbers.	
	uint32_t n_curves;
	uint32_t n_vertices;

	void reserve(int num_curves, int num_vertex) {

		n_curves = num_curves;
		n_vertices = num_vertex;

		vertex.resizeWithoutCopy(n_vertices);
		vertex_path_id.resizeWithoutCopy(n_vertices);

		curve_vertex_pos.resizeWithoutCopy(n_curves);
		curve_type.resizeWithoutCopy(n_curves);
		curve_path_id.resizeWithoutCopy(n_curves);
		curve_contour_id.resizeWithoutCopy(n_curves);
		curve_arc_w.resizeWithoutCopy(n_curves);
		curve_offset.resizeWithoutCopy(n_curves);
		curve_reversed.resizeWithoutCopy(n_curves);

		n_curves = 0;
		n_vertices = 0;
	}

};

struct VGInputPathDataPack {

	// contour
	CUDAArray<uint32_t> contour_first_curve;
	CUDAArray<uint32_t> contour_last_curve;

	// path
	CUDAArray<uint8_t> fill_rule;
	CUDAArray<uint32_t> fill_info;

	CUDAArray<uint8_t> stroke_cap_join; 
	CUDAArray<float> stroke_miter_limit;
	CUDAArray <float> stroke_width;
	CUDAArray<uint32_t> stroke_info;

	// numbers.
	uint32_t n_contours;
	uint32_t n_paths;

	void reserve(int num_paths, int num_contours) {
		n_paths = num_paths;
		n_contours = num_contours;

		contour_first_curve.resizeWithoutCopy(n_contours);
		contour_last_curve.resizeWithoutCopy(n_contours);

		fill_rule.resizeWithoutCopy(num_paths);
		fill_info.resizeWithoutCopy(num_paths);

		stroke_cap_join.resizeWithoutCopy(num_paths);
		stroke_miter_limit.resizeWithoutCopy(num_paths);
		stroke_width.resizeWithoutCopy(num_paths);
		stroke_info.resizeWithoutCopy(num_paths);

		n_paths = 0;
		n_contours = 0;
	}
};

struct TTextInputContext {
	float x, y;
	float h_font;
	int path_index;
	int curve_index;
	int vertex_index;
	int color;
	int ch_last;
};

struct ElementNumber {
	int path;
	int curve;
	int vertex;
	int curve_fragment;
	int merged_fragment;
	int span;
};

class VGRasterizer {

public:
	VGRasterizer();
	virtual ~VGRasterizer();

public:
	virtual void init();
	virtual void uninit() = 0;

	virtual void addVg(const VGContainer &vgc);
	virtual void clear();

	void resizeOutputBuffer(int _widht, int _height);
	glm::ivec2 outputBufferSize() { return _output_buffer_size; }
	void viewport(int x, int y, int _width, int _height);
	void resizeViewport(int _width, int _height);

	virtual void setFragmentSize(int s) = 0;

	virtual void rasterizeImpl() = 0;
	void rasterize();
	void rasterize(bool verbose);

public:
	void setInputTransform(const glm::mat4x4 &mat);
	void setRevTransform(const glm::vec4 &rx, const glm::vec4 &ry, const glm::vec4 rw,
		const glm::vec4 rp, float a);
	void setFPS(bool f) { _show_fps = f; }
	void setAnimation(bool f) { _animation = f; }

	//void setOutput(uint32_t out) { _output = out; }
	void saveOutputToFile(const std::string &fn) {
		_save_output_to_file = true; 
		_output_file_name = fn;
	}
	void verticalFlipOutputFile(bool f) { _vertical_flip_output_file = f; }
	void breakBeforeGL(bool f) { _break_before_gl = f; }
	void countPixel(bool f) { _count_pixel = f; }
	uint64_t pixelCount() { return _pixel_count; }

	void enableOutputTransform(bool f = true) { _enable_output_transform = f; }
	void disableOutputTransform(bool f = true) { _enable_output_transform = !f; }
	void setOutputTransform(const glm::mat4x4 &mat);
	
	void enableSRGBCorrection(bool flag = true) { _enableSRGBCorrection = flag; }

	void verbose(bool v) { _verbose = v; }
	void drawCurve(bool f) { _draw_curve = f; }
	void enableStepTiming(bool f) { _enable_step_timing = f; }
	std::vector<double> stepTiming();
	void dumpDebugData() { _dump_debug_data = true; }

	virtual void setSamples(int s) { _samples = s; }
	virtual void setMultisampleOutput(bool f) { _multisample_output = f; }
	void useMaskTable(bool f) { _use_mask_table = f; }

	void enableStencilBuffer(bool f) { _enable_stencil_buffer = f; }
	void enableTigerClip(bool f) { _enable_tiger_clip = f; }

	ElementNumber elementNumber() { return _element_number; }

public:

	void reserveInk(int s) { 
		_reservedPathNumber = s; 
		_reservedCurveNumber = s * 4;
		_reservedVertexNumber = _reservedCurveNumber * 3;
	}

	void addInk(const std::vector<float2> &vertex, const std::vector<uint32_t> &color, 
		bool newPath = true);
	void startTextInput(int x, int y, float h_font, uint32_t color);
	void onCharMessage(int ch);
	void insertChar(int ch);
	void stopTextInput();
	int isTextInputStarted();

protected:
	virtual void onResize(int width, int height);

protected:
	void unifyCurveVertexOrder();
	void loadMask();

protected:
	int width() { return _output_buffer_size.x; }
	int height() { return _output_buffer_size.y; }

private:
	void initProgram();
	void initTextureAndFramebuffer();

private:
	void drawCurve();
	void drawOutputResize(GLuint ifbo, GLuint ofbo);
	void saveOutputToFile();

protected:
	void cuglUpdateBuffer(int size, NamedBuffer &buffer, cudaGraphicsResource *&res) {
		if (size > buffer.size()) {
			size = (int)(size * 1.5);
			buffer.destroy().create().target(GLPP::Array).storage(size, nullptr,
				GL_MAP_READ_BIT);
			if (res) { cudaGraphicsUnregisterResource(res); res = nullptr; }
			cudaGraphicsGLRegisterBuffer(&res, buffer, cudaGraphicsRegisterFlagsNone);
		}
	}

	void *cuglMap(cudaGraphicsResource *&res) {
		cudaGraphicsMapResources(1, &res, 0);
		void *ptr;
		size_t size;
		cudaGraphicsResourceGetMappedPointer((void**)&ptr, &size, res);
		return ptr;
	}

	void cuglUnMap(cudaGraphicsResource *&res) {
		if (!res) { return; }
		cudaGraphicsUnmapResources(1, &res, 0);
	}

protected:
	std::string tigerClipString();

protected:

	// rasterize config
	glm::ivec2 _vp_pos;
	glm::ivec2 _vp_size;

	glm::ivec2 _output_buffer_size;
	glm::ivec2 _vg_size;

	int _samples = 32;
	bool _use_mask_table = true;
	bool _enable_stencil_buffer = false;
	bool _enable_tiger_clip = false;
	bool _multisample_output = true;
	bool _break_before_gl = false;
	bool _count_pixel = false;
	uint64_t _pixel_count = 0;
	bool _animation = false;
	bool _dump_debug_data = false;

	//
	bool _enableSRGBCorrection = true;
	bool _gpu_stroke_to_fill = false;
	bool _show_fps = false;

	//
	float4 _input_transform[4];
	float _inv_projection_context[13];

	//
	bool _enable_output_transform = false;
	float4 _output_transform[4];

	//
	bool _verbose = false;
	bool _draw_curve = false;
	bool _enable_step_timing = false;
	std::vector<double> _step_timing;
	std::vector<long long> _step_timestamp;

	bool _save_output_to_file = false;
	std::string _output_file_name;
	bool _vertical_flip_output_file = false;

	ElementNumber _element_number;

	// -------- -------- -------- --------
	VGInputCurveDataPack _vg_in_curve[2];
	VGInputPathDataPack _vg_in_path[2];
	int _current_input_pack_id;

	float _gradient_irampheight;
	std::vector<int> _gradient_ramp;
	std::vector<int> _gradient_table;

	int _fps = 0;

	// -------- -------- -------- --------
	// for ink & text

	int _reservedPathNumber = 0;
	int _reservedCurveNumber = 0;
	int _reservedVertexNumber = 0;

	int _nextPathIndex = 0;
	int _nextCurveIndex = 0;
	int _nextVertexIndex = 0;

	int _currentPathId = 0;

	bool _last_one_is_ink = false;

	std::vector<TTextInputContext> _text_input_contexts;
	void* _hfont;

	// -------- -------- -------- --------
	struct {
		CUDAArray<uint32_t> a;
		CUDAArray<uint32_t> p;
	} _mask;

	// -------- -------- -------- --------
	CUDATL::CUDAHostArray<int> _base_host_int;

	CUDAArray<uint64_t> _base_gpu_is_path_visible;

	// -------- -------- -------- --------
	// OpenGL resource

	// buffer
	GLPP::NamedBuffer _base_gl_buffer_gradient_table;

	// texture buffer
	GLPP::NamedTexture _base_gl_texture_gradient_table;

	// texture 2D
	GLPP::NamedTexture _base_gl_texture_gradient_ramp;
	GLPP::NamedTexture _base_gl_texture_output;
	GLPP::NamedTexture _base_gl_texture_output_scale;
	GLPP::NamedTexture _base_gl_texture_output_stencil;
	GLPP::NamedTexture _base_gl_texture_output_integrate_samples;

	// framebuffer
	GLPP::NamedFramebuffer _base_gl_framebuffer_output;
	GLPP::NamedFramebuffer _base_gl_framebuffer_output_scale;
	GLPP::NamedFramebuffer _base_gl_framebuffer_output_integrate_samples;

	// shader program
	GLPP::ShaderProgram _base_gl_program_curve;
	GLPP::ShaderProgram _base_gl_program_output_scale;
	
	GLPP::ShaderProgram _base_gl_program_fps;

	// vertex array
	GLPP::VertexArray _base_gl_vertex_array_empty;

	GLPP::Util _base_gl_util;

	// 
	GLuint _base_gl_tiger_clip_path = 0;
};

} // end of namespace RasterizerBase

} // end of namespace Mochimazui

#endif
