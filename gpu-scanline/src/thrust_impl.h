
#ifndef _MOCHIMAZUI_THRUST_IMPL_H_
#define _MOCHIMAZUI_THRUST_IMPL_H_

#include <cstdint>

namespace Mochimazui {

void thrust_exclusive_scan(int8_t *ibegin, uint32_t number, int8_t *obegin);
void thrust_exclusive_scan(uint8_t *ibegin, uint32_t number, uint8_t *obegin);

void thrust_exclusive_scan(int32_t *ibegin, uint32_t number, int32_t *obegin);
void thrust_exclusive_scan(uint32_t *ibegin, uint32_t number, uint32_t *obegin);

void thrust_exclusive_scan(float *ibegin, uint32_t number, float *obegin);

void thrust_inclusive_scan(int32_t *ibegin, uint32_t number, int32_t *obegin);
void thrust_inclusive_scan(uint32_t *ibegin, uint32_t number, uint32_t *obegin);

}

#endif