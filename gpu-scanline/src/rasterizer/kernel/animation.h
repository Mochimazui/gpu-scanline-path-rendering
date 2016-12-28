
#pragma once

#include <cstdint>
#include "../shared/ras_base.h"

namespace Mochimazui {

void vg_animation(
	int last_frame_timestamp,
	int next_frame_timestamp,
	RasterizerBase::VGInputCurveDataPack &_last_frame_curve_in,
	RasterizerBase::VGInputCurveDataPack &_next_frame_curve_in,
	RasterizerBase::VGInputPathDataPack &_last_frame_path_in,
	RasterizerBase::VGInputPathDataPack &_next_frame_path_in
	);

}
