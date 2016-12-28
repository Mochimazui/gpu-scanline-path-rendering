
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

//#include <gpu/cutil.h>

#include "ras_define.h"
#include "ras_cut.h"

//#define LAUNCH(kernel,N,NT,args) {kernel <<< divup(N,NT),NT >>>args;DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(#kernel);}
//#define GET_ID() (blockDim.x * blockIdx.x + threadIdx.x)

#define DEV static __device__ inline
#define BOTH __device__ __host__ inline

typedef long long i64;
