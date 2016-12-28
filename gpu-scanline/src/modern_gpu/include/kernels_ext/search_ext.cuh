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

#include "../kernels/search.cuh"

#include "../../../cuda/cuda_cached_allocator.h"

namespace mgpu_ext {

	using Mochimazui::g_alloc;

	using namespace mgpu;

	template<MgpuBounds Bounds, typename It1, typename Comp>
	int *BinarySearchPartitions(int count, It1 data_global, int numItems,
		int nv, Comp comp) {

		const int NT = 64;
		int numBlocks = MGPU_DIV_UP(count, nv);
		int numPartitionBlocks = MGPU_DIV_UP(numBlocks + 1, NT);
		//MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numBlocks + 1);
		int *partitionsDevice = (int*)g_alloc.allocate(sizeof(int) * (numBlocks + 1));

		KernelBinarySearch<NT, Bounds>
			<< <numPartitionBlocks, NT >> >(count, data_global,
			numItems, nv, partitionsDevice, numBlocks + 1, comp);
		MGPU_SYNC_CHECK("KernelBinarySearch");

		return partitionsDevice;
	}

	template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
	int *MergePathPartitions_ext(It1 a_global, int aCount, It2 b_global,
		int bCount, int nv, int coop, Comp comp) {

		const int NT = 64;
		int numPartitions = MGPU_DIV_UP(aCount + bCount, nv);
		int numPartitionBlocks = MGPU_DIV_UP(numPartitions + 1, NT);

		//MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);
		int *partitionsDevice = g_alloc.allocate<int>(numPartitions + 1);

		KernelMergePartition<NT, Bounds>
			<< <numPartitionBlocks, NT>> >(a_global, aCount,
			b_global, bCount, nv, coop, partitionsDevice, numPartitions + 1,
			comp);
		MGPU_SYNC_CHECK("KernelMergePartition");

		return partitionsDevice;
	}

}

