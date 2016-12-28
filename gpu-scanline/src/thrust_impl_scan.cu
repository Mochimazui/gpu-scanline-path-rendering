
#include "thrust_impl.h"

#pragma warning( push, 0 ) 
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#pragma warning( pop )

#include "cuda/cuda_cached_allocator.h"

namespace Mochimazui {

// -------- -------- -------- -------- -------- -------- -------- --------
void thrust_exclusive_scan(int8_t *ibegin, uint32_t number, int8_t *obegin) {
	thrust::exclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin, 0);
}

void thrust_exclusive_scan(uint8_t *ibegin, uint32_t number, uint8_t *obegin) {
	thrust::exclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void thrust_exclusive_scan(int32_t *ibegin, uint32_t number, int32_t *obegin) {
	thrust::exclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin, 0);
}

void thrust_exclusive_scan(uint32_t *ibegin, uint32_t number, uint32_t *obegin) {
	thrust::exclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void thrust_exclusive_scan(float *ibegin, uint32_t number, float *obegin) {
	thrust::exclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void thrust_inclusive_scan(int32_t *ibegin, uint32_t number, int32_t *obegin) {
	thrust::inclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin);
}

void thrust_inclusive_scan(uint32_t *ibegin, uint32_t number, uint32_t *obegin) {
	thrust::inclusive_scan(thrust::cuda::par(g_alloc),
		ibegin, ibegin + number, obegin);
}

}
