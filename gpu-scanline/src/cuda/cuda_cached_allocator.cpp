
#include "cuda_cached_allocator.h"

namespace Mochimazui {

cuda_cached_allocator g_thrustCachedAllocator;
cuda_cached_allocator &g_alloc = g_thrustCachedAllocator;

}