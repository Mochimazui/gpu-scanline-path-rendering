
#pragma once

#include "../shared/ras_base.h"

#include <map>

#include <cuda.h>
#include <host_defines.h>

#include "mochimazui/3rd/gl_4_5_compatibility.h"
#include <cuda_gl_interop.h>
#include "mochimazui/glpp.h"
#include "mochimazui/cuda_array.h"

namespace Mochimazui {

class VGContainer;

namespace Rasterizer_R_Cut_A_Mask_Comb_Scanline {

using GLPP::NamedBuffer;
using GLPP::NamedFramebuffer;
using GLPP::NamedTexture;
using GLPP::ShaderProgram;

using CUDATL::CUDAArray;

class VGRasterizer : public RasterizerBase::VGRasterizer {

	typedef RasterizerBase::VGRasterizer _Base;

public:
	VGRasterizer();
	~VGRasterizer();

	void init();
	void uninit();

	void addVg(const VGContainer &vgc);
	void clear() {}

	void setFragmentSize(int s) { _fragSize = s; }

	void rasterizeImpl();

private:
	void initProgram();
	void initBuffer();
	void initFramebuffer();

	void initCommandList();
	void uninitCommandList();

	void onResize(int _width, int _height);

private:
	void initQMMaskTable();

	template <int FRAG_SIZE>
	void rasterizeImpl();

protected:

	uint32_t _fragSize = 2;

	// for debug.
	bool _dbgDumpWindingNumber = false;
	bool _dbgDumpFragmentData = false;

	struct _GL{
		_GL() {}
		struct _GL_Buffer{
			_GL_Buffer() {}

			NamedBuffer stencilDrawData;
			NamedBuffer stencilDrawMask;

			NamedBuffer outputIndex;
			NamedBuffer outputFragmentData;
			NamedBuffer outputSpanData;
			NamedBuffer outputFillInfo;

			NamedBuffer qm_output_stencil_mask;

			// -- debug --
			NamedBuffer dbgCurveVertex;
			NamedBuffer dbgCurveColor;

			NamedBuffer dbgDrawStencilDump_0;
			NamedBuffer dbgDrawStencilDump_1;
			NamedBuffer dbgDrawStencilDump_2;
		} buffer;

		struct _GL_Texture{
			_GL_Texture() {}

			// texbuffer
			NamedTexture stencilDrawData;
			NamedTexture stencilDrawMask;

			NamedTexture outputIndex;
			NamedTexture outputFragmentData;
			NamedTexture outputSpanData;
			NamedTexture outputFillInfo;

			// tex2D
			NamedTexture stencilDraw;

			// -- debug --
			NamedTexture dbgCurveVertex;
			NamedTexture dbgCurveColor;

			NamedTexture dbgDrawCount;

			NamedTexture dbgDrawStencilDump_0;
			NamedTexture dbgDrawStencilDump_1;
			NamedTexture dbgDrawStencilDump_2;
		} texture;

		struct _GL_Framebuffer{
			_GL_Framebuffer() {}
			NamedFramebuffer stencilDrawMS;
		} framebuffer;

		struct _GL_Program{
			_GL_Program() {}

			ShaderProgram output;

			// -- debug --
			ShaderProgram dbgCurve;
			ShaderProgram dbgCurveFragment;
			ShaderProgram dbgOutputScale;

		} program;

	} _gl;

	struct _GPU_Array{
		_GPU_Array() {}

		// transform && stroke to fill
		CUDAArray<float2> strokeTransformedVertex;
		CUDAArray<int> strokeToFillNewCurveTemp;
		
		CUDAArray<float2> transformedVertex;
		
		// monotonize
		CUDAArray<int> curve_pixel_count;
		CUDAArray<float> monotonic_cutpoint_cache;		
		CUDAArray<float> intersection;

		CUDAArray<float> monoCurveT;
		CUDAArray<uint32_t> monoCurveNumber;
		CUDAArray<uint32_t> monoCurveSize;
		CUDAArray<uint32_t> curveFragmentNumber;

		CUDAArray<int32_t> ic4Context;

		CUDAArray<int32_t> fragmentData;

		// mask
		CUDAArray<uint32_t> amaskTable;
		CUDAArray<uint32_t> pmaskTable;

		// temp for CUDA SM gen stencil
		CUDAArray<int32_t> blockBoundaryBins;

		// for CUDA cell list output
		CUDAArray<int32_t> cellListPos;
		CUDAArray<int32_t> cellListFillInfo;
		CUDAArray<int32_t> cellListMaskIndex;

	} _gpu;

	struct __CUDA {
		__CUDA() {}
		struct __CUDAResrouce {
			__CUDAResrouce() :
				stencilDrawData(nullptr), stencilDrawMask(nullptr),
				outputIndex(nullptr), outputFragment(nullptr),
				outputSpan(nullptr), outputFillInfo(nullptr)
			{}

			cudaGraphicsResource *stencilDrawData = nullptr;
			cudaGraphicsResource *stencilDrawMask = nullptr;

			cudaGraphicsResource *outputIndex = nullptr;
			cudaGraphicsResource *outputFragment = nullptr;
			cudaGraphicsResource *outputSpan = nullptr;
			cudaGraphicsResource *outputFillInfo = nullptr;

			cudaGraphicsResource *qm_output_stencil_mask = nullptr;
		} resource;
	} _cuda;

	CUDAArray<int> _qm_mask_table_pixel8;
	CUDAArray<int4> _qm_mask_table_pixel32;

	CUDAArray<float2> _sample_position;
};

} // end of namespace BigFragAM

} // end of namespace Mochimazui
