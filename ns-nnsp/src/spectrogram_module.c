#include "spectrogram_module.h"
#include "ambiq_stdint.h"
#include "ambiq_nnsp_const.h"
#include "ambiq_nnsp_debug.h"
#include "minmax.h"
#if ARM_OPTIMIZED==0
#include "fft.h"
#else
#include "fft_arm.h"
#endif
int32_t glob_fft_input[512 << 1];
int32_t glob_fft_output[512 << 1];

extern const int16_t len_stft_win_coeff;
extern const int16_t hop;
extern const int16_t stft_win_coeff[];
int stftModule_construct(stftModule *ps)
{
	ps->len_win = len_stft_win_coeff;
	ps->hop = hop;
	ps->len_fft = LEN_FFT_NNSP;
	ps->window = stft_win_coeff;
#if ARM_OPTIMIZED==1
	arm_fft_init(ps->fft_st,  0);
	arm_fft_init(ps->ifft_st, 1);
#endif	
	return 0;
}

int stftModule_setDefault(stftModule* ps)
{
	int i;
	for (i = 0; i < ps->len_win; i++)
	{
		ps->dataBuffer[i] = 0;
		ps->odataBuffer[i] = 0;
	}
	return 0;
}

#if ARM_OPTIMIZED==0
void spec2pspec(
		int32_t* y,
		int32_t* x,
		int len)
{
	int i;
	int64_t tmp;
	for (i = 0; i < len; i++)
	{
		tmp = (int64_t)x[2 * i] * (int64_t)x[2 * i] + (int64_t)x[2 * i + 1] * (int64_t)x[2 * i + 1];
		y[i] = (int32_t) (tmp >> 15);
	}
}

int stftModule_analyze(
		stftModule* ps,
		int16_t* x,
		int32_t* y)
{
	int i;
	int32_t tmp;
	static int32_t fft_in[LEN_FFT_NNSP];
	for (i = 0; i < (ps->len_win - ps->hop); i++)
		ps->dataBuffer[i] = ps->dataBuffer[i + ps->hop];

	tmp = ps->len_win - ps->hop;
	for (i = 0; i < ps->hop; i++)
		ps->dataBuffer[i + tmp] = x[i];

	for (i = 0; i < ps->len_win; i++)
	{
		tmp = ((int32_t) ps->window[i] * (int32_t) ps->dataBuffer[i]);
		fft_in[i] = tmp >> 15; // Frac15
	}

	for (i = 0; i < (ps->len_fft - ps->len_win); i++)
	{
		fft_in[i + ps->len_win] = 0;
	}
	rfft(ps->len_fft,
		 fft_in, 
		 (void*)y); //Frac15
	
	return 0;
}
#else
void spec2pspec_arm(
		int32_t* y, // q15
		int32_t* x, // q21
		int len)
{
	int i;
	int64_t tmp;
	int rshift = (21 << 1) - 15;
	for (i = 0; i < len; i++)
	{
		tmp = (int64_t)x[2 * i] * (int64_t)x[2 * i] + (int64_t)x[2 * i + 1] * (int64_t)x[2 * i + 1];
		y[i] = (int32_t) (tmp >> rshift);
	}
}

int stftModule_analyze_arm(
		void* ps_,
		int16_t* input, // q15
		int32_t* output) // q21
{
	int i;
	int32_t tmp;
	stftModule* ps = (stftModule*) ps_;
	
	for (i = 0; i < (ps->len_win - ps->hop); i++)
		ps->dataBuffer[i] = ps->dataBuffer[i + ps->hop];
	
	tmp = ps->len_win - ps->hop;
	for (i = 0; i < ps->hop; i++)
		ps->dataBuffer[i + tmp] = input[i];

	for (i = 0; i < ps->len_win; i++)
	{
		tmp = ((int32_t)ps->window[i] * (int32_t)ps->dataBuffer[i]);
		glob_fft_input[i] = tmp; // Q30
	}

	for (i = 0; i < (ps->len_fft - ps->len_win); i++)
	{
		glob_fft_input[i + ps->len_win] = 0;
	}

	arm_fft_exec(ps->fft_st, output, glob_fft_input); // y: Q21, fft_in: Q30 

	return 0;
}

int stftModule_synthesize_arm(
		void* ps_,
		int32_t* input,
		int32_t* output) 
{
	int i;
	int64_t tmp64;
	int32_t tmp32;
	stftModule* ps = (stftModule*) ps_;

	arm_fft_exec(glob_fft_output, 	// Q21
				 input);			// Q21
	
	for (i = 0; i < ps->len_win; i++)
	{
		tmp64 = ((int64_t) ps->window[i]) * (int64_t) glob_fft_output[i];
		tmp64 >>= 21;
		tmp64 = MIN(MAX(tmp64, INT32_MIN), INT32_MAX);
		tmp32 = (int32_t) ps->odataBuffer[i] + (int32_t) tmp64;

		ps->odataBuffer[i] = MIN(MAX(tmp32, INT16_MIN), INT16_MAX);
	
	}
	
	for (i = 0; i < ps->hop; i++)
		y[i] = ps->odataBuffer[i];

	for (i = 0; i < ps->len_win - ps->hop; i++)
	{
		ps->odataBuffer[i] = ps->odataBuffer[i + ps->hop];
	}

	for (i = 0; i < ps->hop; i++)
	{
		ps->odataBuffer[i + ps->len_win - ps->hop] = 0;
	}

	return 0;
}
#endif
