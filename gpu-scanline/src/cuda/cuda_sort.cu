
#include "cuda_sort.h"

#include <cstdint>

#include <iostream>

#include "../modern_gpu/include/kernels_ext/segmentedsort_ext.cuh"

namespace Mochimazui {

	void cuda_seg_sort_int_by_int(int* key,int* data,int n,int* segs,int nsegs){
		mgpu_ext::SegSortPairsFromIndices(key, data, n, segs,nsegs);
	}

}
