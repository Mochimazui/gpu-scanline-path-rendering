
#ifndef _MOCHIMAZUI_RASTERIZER_SHARED_DEFINE_H_
#define _MOCHIMAZUI_RASTERIZER_SHARED_DEFINE_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <mochimazui/3rd/helper_cuda.h>

// -------- -------- -------- -------- -------- -------- -------- --------
// CONST

//#define VG_RASTERIZER_FRAGMENT_TEXTURE_WIDTH (1024)
//#define VG_RASTERIZER_FRAGMENT_TEXTURE_HEIGHT (1024)

//#define VG_RASTERIZER_FRAGMENT_TEXTURE_WIDTH (2048)
//#define VG_RASTERIZER_FRAGMENT_TEXTURE_HEIGHT (2048)

#define VG_RASTERIZER_FRAGMENT_TEXTURE_WIDTH (4096)
#define VG_RASTERIZER_FRAGMENT_TEXTURE_HEIGHT (4096)

#define VG_RASTERIZER_BIG_FRAGMENT_SIZE (2)

// -------- -------- -------- -------- -------- -------- -------- --------
// CUDA

#define CUDA_DEVICE_SYNC_AND_CHECK_ERROR(msg) { \
	cudaDeviceSynchronize(); __getLastCudaError (msg, __FILE__, __LINE__); }

//
#ifdef _DEBUG
#define DEBUG_CUDA_DEVICE_SYNC() cudaDeviceSynchronize()
#define DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(msg) { \
	cudaDeviceSynchronize(); __getLastCudaError (msg, __FILE__, __LINE__); }
#else
#define DEBUG_CUDA_DEVICE_SYNC()
#define DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(msg)
//#define DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(msg) { \
//	cudaDeviceSynchronize(); __getLastCudaError (msg, __FILE__, __LINE__); }
//#define DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(msg) {timer_print(msg,__FILE__, __LINE__);}
#endif

#define GET_ID() (blockDim.x * blockIdx.x + threadIdx.x)

inline int divup(int a, int b) { return (a + (b - 1)) / b; }
#define LAUNCH(kernel,N,NT,args) {kernel <<< divup(N,NT),NT >>>args; \
    DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(#kernel);}
//#define LAUNCH(kernel,N,NT,args) {kernel <<< (N+(NT-1))/NT,NT >>>args; \
//	DEBUG_CUDA_DEVICE_SYNC_AND_CHECK_ERROR(#kernel);}

#define ASSERT(a)

// -------- -------- -------- -------- -------- -------- -------- --------
// GL 

#ifdef _DEBUG
#define DEBUG_GL_FINISH() glFinish()
#else
#define DEBUG_GL_FINISH()
//#define DEBUG_GL_FINISH() glFinish()
#endif

#define SHADER_DEFINE_TEXT(text) #text
#define SHADER_DEFINE(name) #name" "SHADER_DEFINE_TEXT(name)
#define SHADER_REDEFINE(new_name, old_name) #new_name" "SHADER_DEFINE_TEXT(old_name)

#ifdef __CUDACC__
#ifndef __ldg
#define __ldg(a) (*(a))
#pragma comment( user, "__ldg not defined" ) 
#endif
#endif

// -------- -------- -------- -------- -------- -------- -------- --------
//#define QM_MASK_TABLE_RES 256
//#define QM_MASK_TABLE_PACKING_SCALE 0.5f
//#define QM_MASK_TABLE_N_SAMPLES 128
//#define QM_MASK_TABLE_FETCH_TEST_RES 33

// -------- -------- -------- -------- -------- -------- -------- --------

//#define ENABLE_MPVG_SHIFT
//#define ENABLE_NVPR_SHIFT

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------
// path visible flag:
//
//         y > height
//
//         5 | 6 | 7
// x < 0   3 | x | 4  x > width
//         0 | 1 | 2 
//
//           y < 0
//

// -------- -------- -------- --------
// Big endian
// 012: 11100000 0x0101010000000000 : ~ 0x0000000101010101
// 247: 00101001 0x0000010001000001 : ~ 0x0101000100010100
// 567: 00000111 0x0000000000010101 : ~ 0x0101010101000000
// 035: 10010100 0x0100000100010000 : ~ 0x0001010001000101

//#define PATH_INVISIBLE(mask) ( \
//	(!(mask & 0x0000000101010101)) \
// || (!(mask & 0x0101000100010100))  \
// || (!(mask & 0x0101010101000000))  \
// || (!(mask & 0x0001010001000101))  \
//)

// -------- -------- -------- --------
// little endian
// 012: 11100000 00000111 : ~ 0x0101010101000000
// 247: 00101001 10010100 : ~ 0x0001010001000101
// 567: 00000111 11100000 : ~ 0x0000000101010101
// 035: 10010100 00101001 : ~ 0x0101000100010100

#define PATH_INVISIBLE(mask) ( \
	(!(mask & 0x0101010101000000)) \
 || (!(mask & 0x0001010001000101))  \
 || (!(mask & 0x0000000101010101))  \
 || (!(mask & 0x0101000100010100))  \
)

// -------- -------- -------- --------
#define PATH_VISIBLE(mask) (!(PATH_INVISIBLE(mask)))

#endif
