#ifndef __FFT_ARM_H__
#define __FFT_ARM_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
void arm_fft_init();
void arm_fft_exec(  int32_t *y,   // Q21
                    int32_t *x ); // Q30
#ifdef __cplusplus
}
#endif
#endif