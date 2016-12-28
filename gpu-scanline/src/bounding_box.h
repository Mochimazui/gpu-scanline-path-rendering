#ifndef _MOCHIMAZUI_BOUNDING_BOX_H_
#define _MOCHIMAZUI_BOUNDING_BOX_H_

#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

namespace Mochimazui {

	struct BoundingBoxFloat {

	public:
		__host__ __device__ BoundingBoxFloat() {
			v[0] = make_float2(1e32, 1e32);
			v[1] = -v[0];
		}

		__host__ __device__ void update(const float2 &a) {
			v[0].x = min(v[0].x, a.x);
			v[0].y = min(v[0].y, a.y);
			v[1].x = max(v[1].x, a.x);
			v[1].y = max(v[1].y, a.y);
		}

	public:
		float2 v[2];
	};

	struct BoundingBoxInt {

	public:
		__host__ __device__ BoundingBoxInt() {
			v[0] = make_int2(0x7FFFFFFF, 0x7FFFFFFF);
			v[1] = -v[0];
		}

		__host__ __device__ void update(const int2 &a) {
			v[0].x = min(v[0].x, a.x);
			v[0].y = min(v[0].y, a.y);
			v[1].x = max(v[1].x, a.x);
			v[1].y = max(v[1].y, a.y);
		}

	public:
		int2 v[2];
	};

	typedef BoundingBoxFloat BBoxF;
	typedef BoundingBoxInt BBoxI;

}

#endif