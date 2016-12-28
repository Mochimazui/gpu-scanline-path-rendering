
#pragma once

#include <string>

namespace Mochimazui {

// -------- -------- -------- -------- -------- -------- -------- --------
enum RasterizerPipelineMode {
	PM_Cut_No_AA,
	PM_Cut_Mask_Sample_Scanline,
	PM_Cut_Mask_Pixel_Scanline,
	PM_Cut_Mask_Comb_Scanline,
};

// -------- -------- -------- -------- -------- -------- -------- --------
typedef RasterizerPipelineMode VGPipelineMode;

// -------- -------- -------- -------- -------- -------- -------- --------
inline std::string ras_pipeline_mode_to_string(RasterizerPipelineMode rpm) {	
	if (rpm == PM_Cut_No_AA) {
		return "cut fragment, no AA";
	}
	else if (rpm == PM_Cut_Mask_Sample_Scanline) {
		return "cut fragment, per sample scanline";
	}
	else if (rpm == PM_Cut_Mask_Pixel_Scanline) {
		return "cut fragment, per pixel scanline";
	}
	else if (rpm == PM_Cut_Mask_Comb_Scanline) {
		return "cut fragment, comb scanline";
	}
	else {
		throw std::runtime_error("ras_pipeline_mode_to_string: unsupported pipeline mode");
	}
}

}
