
#ifndef _MOCHIMAZUI_VG_CONFIG_H_
#define _MOCHIMAZUI_VG_CONFIG_H_

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <boost/program_options.hpp>

#include <glm/glm.hpp>

#include <mochimazui/config.h>

#include "rasterizer/shared/ras_pipeline_mode.h"

namespace Mochimazui {

namespace PRIVATE {
extern boost::program_options::variables_map g_config_variables;
}

int init_config(int argc, char *argv[]);

template<class T>
T get_config(const std::string &key) {
	using PRIVATE::g_config_variables;
	if (g_config_variables.count(key)) {
		return g_config_variables[key].as<T>();
	}
	throw std::runtime_error("Mochimazui::get_config: \"" + key + "\" not set");
}

// -------- -------- -------- -------- -------- -------- -------- --------

namespace VGConfig {

// general

inline bool Help() { return get_config<bool>("help"); }

inline bool Verbose() { return get_config<bool>("verbose"); }
inline bool GLDebug() { return get_config<bool>("gl-debug"); }
inline bool DrawCurve() { return get_config<bool>("draw-curve"); }
inline bool ShowFPS() { return get_config<bool>("show-fps"); }

inline bool Benchmark() { return get_config<bool>("benchmark"); }
inline bool StepTiming() { return get_config<bool>("step-timing"); }
inline const std::string AttachTimingToFile() {
	return get_config<std::string>("attach-timing-to");
}

inline bool MergeAdjacentPath() { return get_config<bool>("merge-path"); }
//inline bool MinimalUI() { return get_config<bool>("minimal-ui"); }
inline bool MinimalUI() { return true; }

inline bool OutputVerticalFlip() { return get_config <bool>("v-flip"); }

inline bool Animation() { return get_config<bool>("animation"); }

// input / output

inline std::string Name() { return get_config<std::string>("input-name"); }
inline std::string InputName() { return get_config<std::string>("input-name"); }
inline std::string InputFile() { return get_config<std::string>("input-file"); }
inline int InputWidth() { return get_config<int>("input-width"); }
inline int InputHeight() { return get_config<int>("input-height"); };

inline int WindowWidth() { return get_config<int>("window-width"); }
inline int WindowHeight() { return get_config<int>("window-height"); }
inline glm::ivec2 WindowSize() {
	return glm::ivec2(WindowWidth(), WindowHeight());
}

inline bool FitVGToWindowSize() { return get_config<bool>("fit-to-window"); }
inline bool FitWindowToVGSize() { return get_config<bool>("fit-to-vg"); }

inline bool SaveOutputFile() { return get_config<bool>("save-output-file"); }
inline std::string OutputFile() { return get_config<std::string>("output-file"); }

inline int OutputWidth() { return get_config<int>("output-width"); }
inline int OutputHeight() { return get_config<int>("output-height"); }
inline glm::ivec2 OutputSize() {
	return glm::ivec2(OutputWidth(), OutputHeight());
}
inline bool FixOutputSize() { return get_config<bool>("fix-output-size"); }

// rasterizer config

inline RasterizerPipelineMode PipelineMode() { 
	if (get_config<bool>("c-m-cs")) {
		return PM_Cut_Mask_Comb_Scanline;
	}
	else {
		return PM_Cut_Mask_Comb_Scanline;
	}	
}

inline bool linearRGB() { return get_config<bool>("lrgb"); }
inline bool sRGB() { return get_config<bool>("srgb"); }

inline int Samples() { return get_config<int>("samples"); }
inline bool MultisampleOutput() { return get_config<bool>("ms-output"); }

inline bool UseMaskTable() { return true; }

inline int ReserveInk() { return get_config<int>("reserve-ink"); }
inline bool TigerClip() { return get_config<bool>("tiger-clip"); }

inline bool BreakBeforeGL() { return get_config<bool>("break-before-gl"); }

inline bool A128() { return get_config<bool>("a128"); }

inline bool CountPixel() { return get_config<bool>("count-pixel"); }
inline std::string AttachPixelCountToFile() { return get_config<std::string>("attach-pixel-count-to"); }

}

}

#endif
