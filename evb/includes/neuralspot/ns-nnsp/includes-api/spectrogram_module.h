#ifndef __SPECTROGRM_MODULE_H__
#define __SPECTROGRM_MODULE_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
typedef struct
{
	int16_t len_win;
	int16_t hop;
	int16_t len_fft;
	int16_t dataBuffer[512];
	const int16_t* window;
}stftModule;

int stftModule_construct(stftModule* ps);

int stftModule_setDefault(stftModule* ps);

int stftModule_analyze_arm(	stftModule* ps,
							int16_t* x, 
							int32_t* y,
							void (*pt_arm_fft) (int32_t*, int32_t*));

int stftModule_analyze(	stftModule* ps,
						int16_t* x,
						int32_t* y);

void spec2pspec(int32_t* y, int32_t* x, int len);

#ifdef __cplusplus
}
#endif
#endif