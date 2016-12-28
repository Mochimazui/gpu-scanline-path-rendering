
#pragma once

#ifndef _MOCHIMAZUI_CURA_ARRAY_H_
#define _MOCHIMAZUI_CUDA_ARRAY_H_

#include <cstdio>
#include <cassert>

#include <vector>
#include <stdexcept>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _DEBUG
#define CTL_ASSERT(x) assert((x)==CUDA_SUCCESS)
#else
#define CTL_ASSERT(x) x
#endif

namespace CUDATL {

	/*
	enum ManagedCUDAArraySyncDirection {
	HostToDevice,
	DeviceToHost
	};
	*/

	template<class T>
	struct cuda_array_allocator {
		static cudaError_t malloc(T**p, size_t s) {
			return cudaMalloc(p, s);
		}
		static cudaError_t free(T*p) {
			return cudaFree(p);
		}
	};

	template<class T>
	struct cuda_array_managed_allocator {
		static cudaError malloc(T**p, size_t s) {
			return cudaMallocManaged(p, s);
		}
		static cudaError_t free(T*p) {
			return cudaFree(p);
		}
	};

	template<class T>
	struct cuda_array_host_allocator {
		static cudaError malloc(T**p, size_t s) {
			return cudaMallocHost(p, s);
		}
		static cudaError_t free(T*p) {
			return cudaFreeHost(p);
		}
	};

	// -------- -------- -------- -------- -------- -------- -------- --------
	// @class cuda_array
	template < class T, class allocator = cuda_array_allocator<T> >
	class cuda_array {

	public:
		cuda_array()
			:_size(0), _reservedSize(0), _gpuPointer(nullptr) {
		}

		~cuda_array() {
			clear();
		}

	public:
		size_t size() { return _size; }
		size_t reserved() { return _reservedSize; }

		void malloc(const size_t size) {
			if (_gpuPointer) { 
				cudaFree(_gpuPointer); 
			}
			CTL_ASSERT(allocator::malloc(&_gpuPointer, size * sizeof(T)));
			_size = size;
			_reservedSize = size;
		}

		//
		//void resize(const size_t newSize) {
		//	if (newSize <= _reservedSize) { _size = newSize; return; }
		//	auto oldGPUPointer = _gpuPointer;
		//	auto newReserved = std::max(newSize, _reservedSize);
		//	allocator::malloc(&_gpuPointer, newReserved * sizeof(T));
		//	if (!_gpuPointer) { throw std::runtime_error("cuda_array::resize: out of memory"); }
		//	if (oldGPUPointer) {
		//		auto oldByteSize = _size * sizeof(T);
		//		cudaMemcpy(_gpuPointer, oldGPUPointer, oldByteSize, cudaMemcpyDeviceToDevice);
		//		cudaFree(oldGPUPointer);
		//	}
		//	_size = newSize;
		//	_reservedSize = newReserved;
		//}

		void resizeWithoutCopy(size_t newSize) {
			if (newSize <= _reservedSize) { _size = newSize; return; }
			if (_gpuPointer) {
				cudaFree(_gpuPointer);
				_gpuPointer = nullptr;
			}
			_size = newSize;
			newSize = (size_t)(newSize*1.5);
			allocator::malloc(&_gpuPointer, newSize * sizeof(T));
			if (!_gpuPointer) { throw std::runtime_error("cuda_array::resizWithoutCopy: out of memory"); }
			_reservedSize = newSize;
		}

		//
		void clear() {
			if (_gpuPointer) { 
				allocator::free(_gpuPointer);
			}
			_gpuPointer = nullptr;
			_size = 0;
		}

		// cpu -> gpu
		void set(const std::vector<T> &v) {
			set(v.data(), v.size());
		}

		void set(const T* data, size_t size) {
			if (size > _size) { resizeWithoutCopy(size); }
			CTL_ASSERT(cudaMemcpy(_gpuPointer, data, size*sizeof(T), cudaMemcpyHostToDevice));
		}

		void setAsync(const std::vector<T> &v) {
			setAsync(v.data(), v.size());
		}

		void setAsync(const T* data, size_t size) {
			if (size > _size) { resizeWithoutCopy(size); }
			CTL_ASSERT(cudaMemcpyAsync(_gpuPointer, data, size*sizeof(T), cudaMemcpyHostToDevice));
		}

		// gpu -> cpu
		void get(std::vector<T> &v) {
			v.resize(_size);
			get(v.data(), v.size());
		}

		void get(T* data, size_t size) {
			if (size > _size) { size = _size; }
			CTL_ASSERT(cudaMemcpy(data, _gpuPointer, size*sizeof(T), cudaMemcpyDeviceToHost));
		}

		void getAsync(std::vector<T> &v) {
			v.resize(_size);
			getAsync(v.data(), v.size());
		}

		void getAsync(T* data, size_t size) {
			if (size > _size) { size = _size; }
			CTL_ASSERT(cudaMemcpyAsync(data, _gpuPointer, size*sizeof(T), cudaMemcpyDeviceToHost));
		}

		// cpu -> gpu
		void setValue(int pos, const T &value) {
			if (pos >= _size) { resize(pos + 1); }
			CTL_ASSERT(cudaMemcpy(_gpuPointer + pos, &value, sizeof(T), cudaMemcpyHostToDevice));
		}

		void setValueAsync(int pos, const T &value) {
			if (pos >= _size) { resize(pos + 1); }
			CTL_ASSERT(cudaMemcpyAsync(_gpuPointer + pos, &value, sizeof(T), cudaMemcpyHostToDevice));
		}

		// gpu -> cpu
		T getValue(size_t pos) {
			T value;
			if (pos >= _size) { throw std::runtime_error("index out of range"); }
			CTL_ASSERT(cudaMemcpy(&value, _gpuPointer + pos, sizeof(T), cudaMemcpyDeviceToHost));
			return value;
		}

		T* gpointer() const { return _gpuPointer; }
		T* cpointer() const { return nullptr; }

		T* gptr() const { return _gpuPointer; }
		T* cptr() const { return nullptr; }

	public:
		operator T* () { return _gpuPointer; }

	private:
		size_t _size;
		size_t _reservedSize;
		T *_gpuPointer;
	};

	// !!! If device support cudaMallocManaged, just use cuda_array with cuda_array_managed_allocator
	//template < class T, class  allocator = cuda_array_allocator<T>>
	//class cuda_array_managed : public cuda_array<T> {
	//};

	template<class T>
	using cuda_device_array = cuda_array < T, cuda_array_allocator<T> >;

	template<class T>
	using cuda_managed_array = cuda_array < T, cuda_array_managed_allocator<T> >;

	template<class T>
	using cuda_host_array = cuda_array < T, cuda_array_host_allocator<T> >;

#ifdef _DEBUG
	template<class T>
	using CUDAArray = cuda_managed_array < T >;

	template<class T>
	using CUDAHostArray = cuda_host_array < T >;

	template<class T>
	using CUDAManagedArray = cuda_managed_array < T >;
#else 
	template<class T>
	using CUDAArray = cuda_array < T >;

	template<class T>
	using CUDAHostArray = cuda_host_array < T >;

	template<class T>
	using CUDAManagedArray = cuda_managed_array < T >;
#endif

}

#endif
