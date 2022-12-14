#ifndef __ARM_INTRINSIC_H__
#define __ARM_INTRINSIC_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
int arm_test_se();
int arm_test_s2i(
    void *pt_cntrl_inst, 
    int16_t *pt_data);
uint32_t test_feat();
uint32_t test_fft();
uint32_t test_upload_pc();
// int test_arm_fft();
// int arm_test_nnsp(
//     void *pt_cntrl_inst, 
//     int16_t *pt_data,
//     int16_t *data_buf);
#ifdef __cplusplus
}
#endif
#endif