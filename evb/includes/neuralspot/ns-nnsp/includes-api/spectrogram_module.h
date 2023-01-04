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
} stftModule;

int stftModule_construct(stftModule* ps);

int stftModule_setDefault(stftModule* ps);

int stftModule_analyze_arm(
		void* ps,
		int16_t* x, // q15
		int32_t* y); // q21

int stftModule_analyze(	stftModule* ps,
						int16_t* x,
						int32_t* y);

void spec2pspec(int32_t* y, int32_t* x, int len);
void spec2pspec_arm(int32_t* y, // q15
					int32_t* x, // q21
					int len);
#ifdef __cplusplus
}
#endif
#endif