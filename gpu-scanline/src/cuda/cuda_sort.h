
#pragma once

#include <cstdint>

namespace Mochimazui {
	void cuda_seg_sort_int_by_int(int* key,int* data,int n,int* segs,int nsegs);	
}
