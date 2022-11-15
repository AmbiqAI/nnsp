#ifndef __AFFINE_H__
#define __AFFINE_H__
#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include "activation.h"

int affine_Krows_8x16(
	int16_t dim_output,
	int16_t** pp_output,
	int8_t** pp_kernel,
	int16_t** pp_bias,
	int16_t* input,
	int16_t dim_input,
	int16_t qbit_kernel,
	int16_t qbit_bias,
	int16_t qbit_input,
	int64_t* pt_accum,
	int8_t is_out,
	void* (*act)(void*, int32_t*, int));

int rc_Krows_8x16(int16_t dim_output,
	int16_t** pp_output,
	int8_t** pp_kernel,
	int8_t** pp_kernel_rec,
	int16_t** pp_bias,
	int16_t* input,
	int16_t* input_rec,
	int16_t dim_input,
	int16_t dim_input_rec,
	int16_t qbit_kernel,
	int16_t qbit_bias,
	int16_t qbit_input,
	int16_t qbit_input_rec,
	void* (*act)(void*, int32_t*, int));

int	fc_8x16(
	int16_t* p_output,
	int8_t* p_kernel,
	int8_t* p_kernel_rec,
	int16_t* p_bias,
	int16_t* input,
	int16_t* input_rec,
	int32_t* c_state,
	int16_t dim_output,
	int16_t dim_input,
	int16_t dim_input_rec,
	int16_t qbit_kernel,
	int16_t qbit_bias,
	int16_t qbit_input,
	int16_t qbit_input_rec,
	ACTIVATION_TYPE act_type,
	void* (*act)(void*, int32_t*, int));

int rc_8x16(int16_t* p_output,
	int8_t* p_kernel,
	int8_t* p_kernel_rec,
	int16_t* p_bias,
	int16_t* input,
	int16_t* input_rec,
	int16_t dim_output,
	int16_t dim_input,
	int16_t dim_input_rec,
	int16_t qbit_kernel,
	int16_t qbit_bias,
	int16_t qbit_input,
	int16_t qbit_input_rec,
	ACTIVATION_TYPE act_type,
	void* (*act)(void*, int32_t*, int));

void shift_64b(int64_t* x, int8_t shift, int len);

#ifdef __cplusplus
}
#endif
#endif
