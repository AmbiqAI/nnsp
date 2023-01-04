#ifndef __SPECTROGRM_MODULE_H__
#define __SPECTROGRM_MODULE_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
#include "ambiq_nnsp_debug.h"
#include <arm_math.h>
#include "ambiq_nnsp_const.h"
typedef struct
{
	int16_t len_win;
	int16_t hop;
	int16_t len_fft;
	int16_t dataBuffer[LEN_FFT_NNSP];
	int16_t odataBuffer[LEN_FFT_NNSP];
	const int16_t* window;
	arm_rfft_instance_q31 fft_st;
	arm_rfft_instance_q31 ifft_st;
	int32_t spec[LEN_FFT_NNSP << 1];
}stftModule;

int stftModule_construct(stftModule* ps);

int stftModule_setDefault(stftModule* ps);

#if ARM_OPTIMIZED==0
void spec2pspec(
		int32_t* y, 
		int32_t* x, 
		int len);

int stftModule_analyze(	
		stftModule* ps,
		int16_t* x,
		int32_t* y);
#else
void spec2pspec_arm(
		int32_t* y, // q15
		int32_t* x, // q21
		int len);

int stftModule_analyze_arm(
		void* ps,
		int16_t* input, // q15
		int32_t* output); // q21
int stftModule_synthesize_arm(
		void* ps,
		int16_t* x,
		int32_t* y);
#endif

#ifdef __cplusplus
}
#endif
#endif