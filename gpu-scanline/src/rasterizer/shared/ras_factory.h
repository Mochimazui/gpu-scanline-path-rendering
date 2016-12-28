
#pragma once

#include "ras_base.h"

#include "rasterizer/R_cut_A_mask_comb_scanline/ras_cut_mask_comb_scanline.h"

#ifdef ENABLE_COMPARISON
#include "rasterizer/R_cut_A_none/ras_cut_no_aa.h"
#include "rasterizer/R_cut_A_mask_sample_scanline/ras_cut_mask_sample_scanline.h"
#include "rasterizer/R_cut_A_mask_pixel_scanline/ras_cut_mask_pixel_scanline.h"
#include "rasterizer/c_cs_cuda_cell_list/ras_c_cs_cuda_cell_list.h"
#include "rasterizer/c_cs_gl_cell_list/ras_c_cs_gl_cell_list.h"
#endif

namespace Mochimazui {

inline std::shared_ptr<RasterizerBase::VGRasterizer> createRasterizer(RasterizerPipelineMode rpm) {

	std::shared_ptr<RasterizerBase::VGRasterizer> p_ras;

	if (rpm == PM_Cut_Mask_Comb_Scanline) {
		p_ras.reset(new Rasterizer_R_Cut_A_Mask_Comb_Scanline::VGRasterizer);
	}
#ifdef ENABLE_COMPARISON
#endif
	else {
		throw std::runtime_error("unsupported pipeline mode");
	}

	return p_ras;
}

}
