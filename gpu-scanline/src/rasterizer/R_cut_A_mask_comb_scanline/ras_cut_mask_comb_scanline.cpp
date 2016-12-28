
#include "ras_cut_mask_comb_scanline.h"

#include <functional>
#include <algorithm>

#include "mochimazui/3rd/helper_cuda.h"
#include "mochimazui/stdio_ext.h"

#include "cuda/cuda_cached_allocator.h"

#include "../shared/ras_define.h"
#include "../../vg_container.h"

#define NO_GL_EXT

namespace Mochimazui {

namespace Rasterizer_R_Cut_A_Mask_Comb_Scanline {

using stdext::error_printf;

// -------- -------- -------- -------- -------- -------- -------- --------
VGRasterizer::VGRasterizer() {
}

VGRasterizer::~VGRasterizer() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::init() {

	_Base::init();

	initProgram();
	initBuffer();
	initFramebuffer();
	initCommandList();

	g_alloc.reserver(256 * 1024 * 1024);

	loadMask();
	initQMMaskTable();

}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::initProgram() {

	auto loadShader = [](GLPP::ShaderType type, const std::string &shaderName) {

		std::shared_ptr<GLPP::Shader> ps(new GLPP::Shader(type));
		ps->name(shaderName);
		ps->define(SHADER_REDEFINE(TEXTURE_WIDTH, VG_RASTERIZER_FRAGMENT_TEXTURE_WIDTH));
		ps->define(SHADER_REDEFINE(TEXTURE_HEIGHT, VG_RASTERIZER_FRAGMENT_TEXTURE_HEIGHT));
		ps->define(SHADER_REDEFINE(BIG_FRAG_SIZE, VG_RASTERIZER_BIG_FRAGMENT_SIZE));

		std::string shaderFile = "./shader/R_cut_A_stencil/" + shaderName
			+ GLPP::Shader::DefaultFileExtension(type);

		ps->codeFromFile(shaderFile);
		return ps;
	};

	auto setShader = [&](GLPP::ShaderProgram &p, const std::string &shaderName) {
		p.name(shaderName)
			.setShader(loadShader(GLPP::Vertex, shaderName))
			.setShader(loadShader(GLPP::Fragment, shaderName))
			.link();
	};

	if (_multisample_output) {
		if (_samples == 8) { setShader(_gl.program.output, "ms_output_8"); }
		else { setShader(_gl.program.output, "ms_output_32"); }
	}
	else {
		if (_samples == 8) { setShader(_gl.program.output, "output_8"); }
		else { setShader(_gl.program.output, "output_32"); }
	}

	CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::initBuffer() {

#ifndef NO_GL_EXT
	static float v[] = {
		0.f, 0.f, 512.f, 512.f
	};
	glCreateBuffers(1, &_glRes.dbgVBO);
	glNamedBufferStorage(_glRes.dbgVBO, sizeof(float) * 4, v, 0);
	glGetNamedBufferParameterui64vNV(_glRes.dbgVBO, GL_BUFFER_GPU_ADDRESS_NV, &_glResPtr.dbgVBO);
	glMakeNamedBufferResidentNV(_glRes.dbgVBO, GL_READ_ONLY);

	static float vp[2] = {
		1024, 1024,
	};
	glCreateBuffers(1, &_glRes.dbgUBO);
	glNamedBufferStorage(_glRes.dbgUBO, sizeof(float) * 2, vp, 0);
	glGetNamedBufferParameterui64vNV(_glRes.dbgUBO, GL_BUFFER_GPU_ADDRESS_NV, &_glResPtr.dbgUBO);
	glMakeNamedBufferResidentNV(_glRes.dbgUBO, GL_READ_ONLY);
#endif

	// -------- -------- -------- --------
	// buffer
	_gl.buffer.stencilDrawData.create();
	_gl.buffer.stencilDrawMask.create();

	_gl.buffer.outputIndex.create();
	_gl.buffer.outputFragmentData.create();
	_gl.buffer.outputSpanData.create();
	_gl.buffer.outputFillInfo.create();

	_gl.buffer.dbgCurveVertex.create();
	_gl.buffer.dbgCurveColor.create();

	if (_dbgDumpFragmentData) {
		_gl.buffer.dbgDrawStencilDump_0.create().storage(sizeof(float) * 4 * 1024 * 512 * 32, 0,
			GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
		_gl.buffer.dbgDrawStencilDump_1.create().storage(sizeof(float) * 4 * 1024 * 512 * 32, 0,
			GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
		_gl.buffer.dbgDrawStencilDump_2.create().storage(sizeof(float) * 4 * 1024 * 512 * 32, 0,
			GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
	}
	else {
		_gl.buffer.dbgDrawStencilDump_0.create().storage(sizeof(float) * 1, 0,
			GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
		_gl.buffer.dbgDrawStencilDump_1.create().storage(sizeof(float) * 1, 0,
			GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
		_gl.buffer.dbgDrawStencilDump_2.create().storage(sizeof(float) * 1, 0,
			GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
	}

	CHECK_GL_ERROR();

	// -------- -------- -------- --------
	// texture used as TextureBuffer.
	_gl.texture.stencilDrawData.target(GLPP::TextureBuffer).create();
	_gl.texture.stencilDrawMask.target(GLPP::TextureBuffer).create();

	_gl.texture.outputIndex.target(GLPP::TextureBuffer).create();
	_gl.texture.outputFragmentData.target(GLPP::TextureBuffer).create();
	_gl.texture.outputSpanData.target(GLPP::TextureBuffer).create();
	_gl.texture.outputFillInfo.target(GLPP::TextureBuffer).create();
	
	CHECK_GL_ERROR();

	// dbg
	_gl.texture.dbgCurveVertex.target(GLPP::TextureBuffer).create();
	_gl.texture.dbgCurveColor.target(GLPP::TextureBuffer).create();

	_gl.texture.dbgDrawStencilDump_0.target(GLPP::TextureBuffer).create()
		.buffer(GL_RGBA32F, _gl.buffer.dbgDrawStencilDump_0);
	_gl.texture.dbgDrawStencilDump_1.target(GLPP::TextureBuffer).create()
		.buffer(GL_RGBA32F, _gl.buffer.dbgDrawStencilDump_1);
	_gl.texture.dbgDrawStencilDump_2.target(GLPP::TextureBuffer).create()
		.buffer(GL_RGBA32F, _gl.buffer.dbgDrawStencilDump_2);

	_gl.texture.dbgDrawCount.target(GLPP::Texture2D).create()
		.storage2D(1, GL_R32I, 1024, 1024);

	CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::initFramebuffer() {

	CHECK_GL_ERROR();

	//
	_gl.texture.stencilDraw.target(GLPP::Texture2DMultisample).create();
	_gl.texture.stencilDraw.storage2DMultisample(32,
		//GL_RG8,
		//GL_RG16F,
		GL_R16F,
		VG_RASTERIZER_FRAGMENT_TEXTURE_WIDTH,
		VG_RASTERIZER_FRAGMENT_TEXTURE_HEIGHT, GL_TRUE);
	_gl.framebuffer.stencilDrawMS.create()
		.texture2D(GL_COLOR_ATTACHMENT0, _gl.texture.stencilDraw, 0);

#ifndef NO_GL_EXT
	if (_vpSize.x > 0 && _vpSize.y > 0) {

		glGenTextures(1, &_glRes.dbgColorTexture);
		glBindTexture(GL_TEXTURE_2D, _glRes.dbgColorTexture);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1024, 1024);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenFramebuffers(1, &_glRes.dbgFramebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, _glRes.dbgFramebuffer);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
			_glRes.dbgColorTexture, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		_glResPtr.dbgColorTexture = glGetTextureHandleARB(_glRes.dbgColorTexture);
		glMakeTextureHandleResidentARB(_glResPtr.dbgColorTexture);

		CHECK_GL_ERROR();
	}
#endif

	CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::initCommandList() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::uninit() {}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::onResize(int width, int height) {

	_Base::onResize(width, height);

	// update shader uniform.
	_gl.program.output.uniform2i("vp_size", width, height);
	CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGRasterizer::addVg(const VGContainer &vgc) {
	_Base::addVg(vgc);
	if (!_gpu_stroke_to_fill) {
		unifyCurveVertexOrder();
	}
}

int tiger_transform_x = 170;
int tiger_transform_y = -150;

} // end of namespace BigFragAM

} // end of namespace Mochimazui
