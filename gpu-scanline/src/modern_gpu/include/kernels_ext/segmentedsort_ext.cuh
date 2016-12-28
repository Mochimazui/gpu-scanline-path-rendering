/******************************************************************************
* Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

/******************************************************************************
*
* Code and text by Sean Baxter, NVIDIA Research
* See http://nvlabs.github.io/moderngpu for repository and documentation.
*
******************************************************************************/

#pragma once

#include "../kernels/segmentedsort.cuh"

#include "search_ext.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

//#include "thrust/thrust_impl.h"
#include "../../../cuda/cuda_cached_allocator.h"

namespace mgpu_ext {

	using Mochimazui::g_alloc;

	using namespace mgpu;

	////////////////////////////////////////////////////////////////////////////////
	// AllocSegSortBuffers

	static inline byte *AllocSegSortBuffers(int count, int nv,
		SegSortSupport& support, bool segments) {

		int numBlocks = MGPU_DIV_UP(count, nv);
		int numPasses = FindLog2(numBlocks, true);
		int numRanges = 1;
		int numBlocks2 = MGPU_DIV_UP(numBlocks, 2);
		for (int pass = 1; pass < numPasses; ++pass) {
			numRanges += numBlocks2;
			numBlocks2 = MGPU_DIV_UP(numBlocks2, 2);
		}

		int rangesSize = MGPU_ROUND_UP_POW2(sizeof(int) * numBlocks, 128);
		int ranges2Size = MGPU_ROUND_UP_POW2(sizeof(int2) * numRanges, 128);
		int mergeListSize = MGPU_ROUND_UP_POW2(sizeof(int4) * numBlocks, 128);
		int copyListSize = MGPU_ROUND_UP_POW2(sizeof(int) * numBlocks, 128);
		int countersSize = MGPU_ROUND_UP_POW2(sizeof(int4), 128);
		int copyStatusSize = MGPU_ROUND_UP_POW2(sizeof(byte) * numBlocks, 128);

		if (!segments) rangesSize = ranges2Size = 0;

		int total = rangesSize + ranges2Size + mergeListSize + copyListSize +
			countersSize + copyStatusSize;

		//MGPU_MEM(byte) mem = context.Malloc<byte>(total);
		byte *mem = (byte*)g_alloc.allocate(total * sizeof(byte));

		if (segments) {
			support.ranges_global = PtrOffset((int*)mem, 0);
			support.ranges2_global = PtrOffset((int2*)support.ranges_global,
				rangesSize);
			support.mergeList_global = PtrOffset((int4*)support.ranges2_global,
				ranges2Size);
		}
		else {
			support.ranges_global = 0;
			support.ranges2_global = 0;
			support.mergeList_global = (int4*)mem;
		}

		support.copyList_global = PtrOffset((int*)support.mergeList_global,
			mergeListSize);
		support.queueCounters_global = PtrOffset((int2*)support.copyList_global,
			copyListSize);
		support.nextCounters_global = PtrOffset(support.queueCounters_global,
			sizeof(int2));
		support.copyStatus_global = PtrOffset((byte*)support.queueCounters_global,
			countersSize);

		// Fill the counters with 0s on the first run.
		cudaMemsetAsync(support.queueCounters_global, 0, sizeof(int4));

		return mem;
	}

	////////////////////////////////////////////////////////////////////////////////
	// SegSortPasses
	// Multi-pass segmented mergesort process. Factored out to allow simpler 
	// specialization over head flags delivery in blocksort.

	template<typename Tuning, bool Segments, bool HasValues, typename KeyType,
		typename ValType, typename Comp>
		MGPU_HOST void SegSortPasses(SegSortSupport& support,
		KeyType* keysSource_global, ValType* valsSource_global,
		int count, int numBlocks, int numPasses, KeyType* keysDest_global,
		ValType* valsDest_global, Comp comp) {

		//int2 launch = Tuning::GetLaunchParams(context);
		int2 launch;
		launch.x = Tuning::Sm30::NT;
		launch.y = Tuning::Sm30::VT;
		int NV = launch.x * launch.y;

		const int NT2 = 64;
		int numPartitions = numBlocks + 1;
		int numPartBlocks = MGPU_DIV_UP(numPartitions, NT2 - 1);
		int numCTAs = min(numBlocks, 16 * 6);
		int numBlocks2 = MGPU_DIV_UP(numBlocks, 2);

		SegSortPassInfo info(numBlocks);
		for (int pass = 0; pass < numPasses; ++pass) {
			if (0 == pass) {
				KernelSegSortPartitionBase<NT2, Segments>
					<< <numPartBlocks, NT2 >> >(keysSource_global,
					support, count, NV, numPartitions, comp);
				MGPU_SYNC_CHECK("KernelSegSortPartitionBase");
			}
			else {
				KernelSegSortPartitionDerived<NT2, Segments>
					<< <numPartBlocks, NT2 >> >(keysSource_global,
					support, count, numBlocks2, pass, NV, numPartitions, comp);
				MGPU_SYNC_CHECK("KernelSegSortPartitionDerived");

				support.ranges2_global += numBlocks2;
				numBlocks2 = MGPU_DIV_UP(numBlocks2, 2);
			}

			KernelSegSortMerge<Tuning, Segments, HasValues>
				<< <numCTAs, launch.x >> >(keysSource_global,
				valsSource_global, support, count, pass, keysDest_global,
				valsDest_global, comp);
			MGPU_SYNC_CHECK("KernelSegSortMerge");

			std::swap(keysDest_global, keysSource_global);
			std::swap(valsDest_global, valsSource_global);
			std::swap(support.queueCounters_global, support.nextCounters_global);
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	// Segmented sort from head flags passed as indices

	template<typename T>
	MGPU_HOST void SegSortKeysFromIndices(T* data_global, int count,
		const int* indices_global, int indicesCount) {

		const bool Stable = true;
		typedef LaunchBoxVT<
			128, 11, 0,
			128, 11, 0,
			128, 11, 0
	> Tuning;

		//int2 launch = Tuning::GetLaunchParams(context);
		int2 launch;
		launch.x = 128;
		launch.y = 11;

		const int NV = launch.x * launch.y;

		int numBlocks = MGPU_DIV_UP(count, NV);
		int numPasses = FindLog2(numBlocks, true);

		SegSortSupport support;
		//MGPU_MEM(byte) mem = AllocSegSortBuffers(count, NV, support, true, context);
		byte *mem = AllocSegSortBuffers(count, NV, support, true);
		
		//MGPU_MEM(T) destDevice = context.Malloc<T>(count);
		T* destDevice = (T*)g_alloc.allocate(count * sizeof(T));

		T* source = data_global;
		T* dest = destDevice;

		//MGPU_MEM(int) partitionsDevice = BinarySearchPartitions<MgpuBoundsLower>(
		//	count, indices_global, indicesCount, NV, mgpu::less<int>(), context);
		int *partitionsDevice = BinarySearchPartitions<MgpuBoundsLower>(
			count, indices_global, indicesCount, NV, mgpu::less<int>());

		KernelSegBlocksortIndices<Tuning, Stable, false>
			<< <numBlocks, launch.x >> >(source, (const int*)0,
			count, indices_global, partitionsDevice,
			(1 & numPasses) ? dest : source, (int*)0, support.ranges_global, mgpu::less<T>());
		MGPU_SYNC_CHECK("KernelSegBlocksortIndices");

		if (1 & numPasses) std::swap(source, dest);

		SegSortPasses<Tuning, true, false>(support, source, (int*)0, count,
			numBlocks, numPasses, dest, (int*)0, mgpu::less<T>());
	}

	template<typename KeyType, typename ValType>
	MGPU_HOST void SegSortPairsFromIndices(KeyType* keys_global,
		ValType* values_global, int count, const int* indices_global,
		int indicesCount) {

		const bool Stable = true;
		typedef LaunchBoxVT<
			128, 11, 0,
			128, 7, 0,
			128, 7, 0
		> Tuning;
		//int2 launch = Tuning::GetLaunchParams(context);
		int2 launch;
		launch.x = Tuning::Sm30::NT;
		launch.y = Tuning::Sm30::VT;
		const int NV = launch.x * launch.y;

		int numBlocks = MGPU_DIV_UP(count, NV);
		int numPasses = FindLog2(numBlocks, true);

		SegSortSupport support;
		byte* mem = AllocSegSortBuffers(count, NV, support, true);

		//MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
		//MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(count);
		KeyType* keysDestDevice = (KeyType*)g_alloc.allocate(sizeof(KeyType)*count);
		ValType* valsDestDevice = (ValType*)g_alloc.allocate(sizeof(ValType)*count);

		KeyType* keysSource = keys_global;
		KeyType* keysDest = keysDestDevice;
		ValType* valsSource = values_global;
		ValType* valsDest = valsDestDevice;

		int* partitionsDevice = BinarySearchPartitions<MgpuBoundsLower>(
			count, indices_global, indicesCount, NV, mgpu::less<int>());

		KernelSegBlocksortIndices<Tuning, Stable, true>
			<< <numBlocks, launch.x >> >(keysSource, valsSource,
			count, indices_global, partitionsDevice,
			(1 & numPasses) ? keysDest : keysSource,
			(1 & numPasses) ? valsDest : valsSource, support.ranges_global, 
			mgpu::less<KeyType>());
		MGPU_SYNC_CHECK("KernelSegBlocksortIndices");

		if (1 & numPasses) {
			std::swap(keysSource, keysDest);
			std::swap(valsSource, valsDest);
		}

		SegSortPasses<Tuning, true, true>(support, keysSource, valsSource, count,
			numBlocks, numPasses, keysDest, valsDest, mgpu::less<KeyType>());
	}

	///////////////////////////////////////////////////////////////////////////
	template<typename T>
	MGPU_HOST void SegSortKeysFromFlags(T* data_global, int count,
		const uint* flags_global) {
	
		const bool Stable = true;
		typedef LaunchBoxVT<
			128, 11, 0,
			128, 11, 0,
			128, (sizeof(T) > 4) ? 7 : 11, 0
		> Tuning;
		int2 launch;
		launch.x = Tuning::Sm30::NT;
		launch.y = Tuning::Sm30::VT;
		const int NV = launch.x * launch.y;
		
		int numBlocks = MGPU_DIV_UP(count, NV);
		int numPasses = FindLog2(numBlocks, true);
		
		SegSortSupport support;
		byte* mem = AllocSegSortBuffers(count, NV, support, true);
	
		//MGPU_MEM(T) destDevice = context.Malloc<T>(count);
		T* destDevice = (T*)g_alloc.allocate(count * sizeof(T));
		T* source = data_global;
		T* dest = destDevice;
	
		KernelSegBlocksortFlags<Tuning, Stable, false>
			<<<numBlocks, launch.x, 0>>>(source, (const int*)0,
			count, flags_global, (1 & numPasses) ? dest : source, (int*)0,
			support.ranges_global, mgpu::less<T>());
		MGPU_SYNC_CHECK("KernelSegBlocksortFlags");
	
		if(1 & numPasses) std::swap(source, dest);
	
		SegSortPasses<Tuning, true, false>(support, source, (int*)0, count, 
			numBlocks, numPasses, dest, (int*)0, mgpu::less<T>());
	}
	
	template<typename KeyType, typename ValType>
	MGPU_HOST void SegSortPairsFromFlags(KeyType* keys_global,
		ValType* values_global, int count, const uint* flags_global) {
	
		const bool Stable = true;
		typedef LaunchBoxVT<
			128, 11, 0,
			128, 7, 0,
			128, 7, 0
		> Tuning;
		int2 launch;
		launch.x = Tuning::Sm30::NT;
		launch.y = Tuning::Sm30::VT;
		const int NV = launch.x * launch.y;
		
		int numBlocks = MGPU_DIV_UP(count, NV);
		int numPasses = FindLog2(numBlocks, true);
	
		SegSortSupport support;
		byte* mem = AllocSegSortBuffers(count, NV, support, true);
	
		//MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
		//MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(count);
		KeyType* keysDestDevice = (KeyType*)g_alloc.allocate(sizeof(KeyType)*count);
		ValType* valsDestDevice = (ValType*)g_alloc.allocate(sizeof(ValType)*count);

		KeyType* keysSource = keys_global;
		KeyType* keysDest = keysDestDevice;
		ValType* valsSource = values_global;
		ValType* valsDest = valsDestDevice;
	
		KernelSegBlocksortFlags<Tuning, Stable, true>
			<<<numBlocks, launch.x, 0>>>(keysSource, valsSource,
			count, flags_global, (1 & numPasses) ? keysDest : keysSource,
			(1 & numPasses) ? valsDest : valsSource, support.ranges_global, 
			mgpu::less<KeyType>());
		MGPU_SYNC_CHECK("KernelSegBlocksortFlags");
	
		if(1 & numPasses) {
			std::swap(keysSource, keysDest);
			std::swap(valsSource, valsDest);
		}
	
		SegSortPasses<Tuning, true, true>(support, keysSource, valsSource, count, 
			numBlocks, numPasses, keysDest, valsDest, mgpu::less<KeyType>());
	}
	
}
