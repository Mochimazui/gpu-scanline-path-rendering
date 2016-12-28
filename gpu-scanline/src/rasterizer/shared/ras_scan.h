
#ifndef _MOCHIMAZUI_RASTERIZER_SHARED_SCAN_H_
#define _MOCHIMAZUI_RASTERIZER_SHARED_SCAN_H_

#include <cuda_runtime.h>
#include "thrust_impl.h"

namespace Mochimazui {

namespace Rasterizer {

inline void escan_with_ret(int* p, int n, int *ret) {
	thrust_exclusive_scan((uint32_t*)p, n + 1, (uint32_t*)p);
	cudaMemcpy(ret, p + n, sizeof(int), cudaMemcpyDeviceToHost);
}

inline int escan(int* p, int n) {
	int ret = 0;
	thrust_exclusive_scan((uint32_t*)p, n + 1, (uint32_t*)p);
	cudaMemcpy(&ret, p + n, sizeof(int), cudaMemcpyDeviceToHost);
	return ret;
}

inline int escan(int* i, int *o, int n) {
	int ret = 0;
	thrust_exclusive_scan((uint32_t*)i, n + 1, (uint32_t*)o);
	cudaMemcpy(&ret, o + n, sizeof(int), cudaMemcpyDeviceToHost);
	return ret;
}

inline int iscan(int* i, int *o, int n) {
	int ret = 0;
	thrust_inclusive_scan((uint32_t*)i, n, (uint32_t*)o);
	cudaMemcpy(&ret, o + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return ret;
}

} // end of namespace Rasterizers

} // end of namespace Mochimazui

#endif