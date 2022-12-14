#include <stdint.h>
#include <arm_math.h>
#include "fft_arm.h"
#include "ambiq_nnsp_const.h"
arm_rfft_instance_q31 fft_st;
void arm_fft_init()
{
    uint32_t 	ifftFlagR = 0;
    uint32_t 	bitReverseFlag=1;
    arm_rfft_init_q31(  &fft_st,
                        LEN_FFT_NNSP, 
                        ifftFlagR, 
                        bitReverseFlag);
}

void arm_fft_exec(  int32_t *y,  // Q21
                    int32_t *x ) // Q30
{
    arm_rfft_q31(&fft_st, x, y);
}
