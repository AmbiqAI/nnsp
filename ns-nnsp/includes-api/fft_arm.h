#ifndef __FFT_ARM_H__
#define __FFT_ARM_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
void arm_fft_init(
        arm_rfft_instance_q31 *p_fft_st,
        uint32_t is_ifft);
void arm_fft_exec(  
        arm_rfft_instance_q31 *p_fft_st,
        int32_t *y,  // Q21
        int32_t *x ); // Q30

#ifdef __cplusplus
}
#endif
#endif