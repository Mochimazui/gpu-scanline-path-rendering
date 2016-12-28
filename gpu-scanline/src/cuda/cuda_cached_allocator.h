
#ifndef _MOCHIMAZUI_THRUST_CACHED_ALLOCATOR_H_
#define _MOCHIMAZUI_THRUST_CACHED_ALLOCATOR_H_

#include <iostream>
#include <exception>
#include <stdexcept>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/system/cuda/vector.h>
//#include <thrust/system/cuda/execution_policy.h>
//#include <thrust/host_vector.h>
//#include <thrust/pair.h>

namespace Mochimazui {

class cuda_cached_allocator_bad_alloc : public std::runtime_error {
public:
	cuda_cached_allocator_bad_alloc(const char *msg) :runtime_error(msg) {}
	cuda_cached_allocator_bad_alloc(const std::string &msg) :runtime_error(msg) {}
};

// Example by Nathan Bell and Jared Hoberock
// (modified by Mihail Ivakhnenko)
//
// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::reduce. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::cuda::malloc.
//
// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

// cached_allocator: a simple allocator for caching allocation requests
class cuda_cached_allocator {
public:
	// just allocate bytes
	typedef char value_type;

	cuda_cached_allocator() {}

	~cuda_cached_allocator() {
		// free all allocations when cached_allocator goes out of scope
		free_all();
	}

public:
	void reserver(size_t s) {
		if (_ptr) { 
			cudaFree(_ptr); 
		}
		cudaMalloc(&_ptr, s);
		_reservedSize = s;
	}
	 
	size_t reserved() {
		return _reservedSize;
	}

	void reset() {
		_unallocatedPtr = 0;
	}

	void fill_zero() {
		cudaMemsetAsync(_ptr, 0, _reservedSize);
	}

	size_t allocated() {
		return _unallocatedPtr;
	}

	char* allocate(std::ptrdiff_t num_bytes) {
		size_t newPtr = _unallocatedPtr + num_bytes;
		if (newPtr > _reservedSize) {
			printf("cuda_cached_allocator: reserved memory exhausted.");
			throw std::runtime_error("cuda_cached_allocator: reserved memory exhausted.");
		}
		char *a = _ptr + _unallocatedPtr;
		_unallocatedPtr = newPtr;

		// 256 bit align
		if (_unallocatedPtr & 0x1F) {
			_unallocatedPtr += (32 - _unallocatedPtr & 0x1F);
		}
		return a;
	}

	template<class T>
	T *allocate(size_t num) {
		return (T*)this->allocate(num *sizeof(T));
	}

	template<class T>
	void allocate(T **ptr, size_t num) {
		*ptr = (T*)this->allocate(num *sizeof(T));
	}

	void deallocate(char* ptr, size_t n) {}

private:
	size_t _reservedSize = 0;
	size_t _unallocatedPtr = 0;
	char *_ptr = nullptr;

private:
	void free_all() {
		cudaFree(_ptr);
	}

};

extern cuda_cached_allocator g_thrustCachedAllocator;
extern cuda_cached_allocator &g_alloc;

}

#endif
