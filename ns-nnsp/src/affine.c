#include "ambiq_stdint.h"
#include "minmax.h"
#include "activation.h"
#if DEBUG_PRINT
#include "extern_files.h"
#endif
#include "affine.h"


int64_t accumulators[4] = { 0, 0, 0, 0 };
void shift_64b(int64_t* x, int8_t shift, int len)
{
	int i;
	int64_t M, m;
	if (shift == 0)
	{

	}
	else if (shift < 0)
	{
		shift = -shift;
		for (i = 0; i < len; i++)
			x[i] >>= shift;
	}
	else
	{
		M = (int64_t)1 << (63 - shift);
		m = -M;
		M -= 1;
		for (i = 0; i < len; i++)
		{
			x[i] = MIN(MAX(x[i], m), M);
			x[i] <<= shift;
		}

	}
}

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
	void* (*act)(void*, int32_t*, int))
{
	int8_t* p_kernel = *pp_kernel;
	int16_t* p_bias = *pp_bias;
	int16_t* po = *pp_output;
	int16_t* pi = input;
	int32_t nbit_out = 32;
	int16_t in[2];
	int32_t acc32[4];
	int i, j;
	int shift;
	int qbit_s;
	if (p_bias == 0)
		qbit_s = qbit_input + qbit_kernel;
	else
		qbit_s = MAX(15, qbit_input + qbit_kernel);


	for (i = 0; i < (dim_input >> 1); i++)
	{
		in[0] = *pi++;
		in[1] = *pi++;
		for (j = 0; j < dim_output; j++)
		{
			pt_accum[j] += (int64_t)p_kernel[0] * (int64_t)in[0] + (int64_t)p_kernel[1] * (int64_t)in[1];
			p_kernel += 2;
		}

	}

	if (dim_input % 2)
	{
		in[0] = *pi++;
		for (j = 0; j < dim_output; j++)
			pt_accum[j] += (int64_t)*p_kernel++ * (int64_t)in[0];

	}

	shift = qbit_s - (qbit_input + qbit_kernel);
	shift_64b(pt_accum, shift, dim_output); // align acc to w

	
	if (p_bias != 0)
	{
		shift = qbit_s - (qbit_bias);
		// align w to acc & add
		for (i = 0; i < dim_output; i++)
		{
			pt_accum[i] += (shift >= 0) ? ((int64_t)*p_bias++) << shift : ((int64_t)*p_bias++) >> -shift;
		}
	}
	

	if (is_out)
	{
		shift = 15 - qbit_s;
		shift_64b(pt_accum, shift, dim_output);
		for (i = 0; i < dim_output; i++)
		{
			acc32[i] = (int32_t)MIN(MAX(pt_accum[i], MIN_INT32_T), MAX_INT32_T);
		}

		po = *pp_output;
		po = (int16_t*)(*act)(po, acc32, dim_output);
		*pp_output = po;

	}
	*pp_kernel = p_kernel;
	*pp_bias = p_bias;

	return 0;

}

int rc_Krows_8x16(	
	int16_t dim_output,
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
	void* (*act)(void*, int32_t*, int))
{
	int16_t* p_output = *pp_output;
	int8_t* p_kernel = *pp_kernel;
	int8_t* p_kernel_rec = *pp_kernel_rec;
	int16_t* p_bias = *pp_bias;
	int16_t* p_bias_null = (int16_t*) 0;
	int8_t is_out;
	int j;
	int shift = qbit_input_rec - qbit_input;

	for (j = 0; j < dim_output; j++)
		accumulators[j] = 0;

	is_out = 0;
	affine_Krows_8x16(dim_output, &p_output, &p_kernel, &p_bias_null, input, dim_input,
		qbit_kernel,
		qbit_bias,
		qbit_input,
		accumulators,
		is_out,
		act);
	shift_64b(accumulators, shift, dim_output);

	is_out = 1;
	affine_Krows_8x16(dim_output, &p_output, &p_kernel_rec, &p_bias, input_rec, dim_input_rec,
		qbit_kernel,
		qbit_bias,
		qbit_input_rec,
		accumulators,
		is_out,
		act);
#if DEBUG_PRINT
	if (is_out)
	{
		for (j = 0; j < rows; j++)
			fprintf(file_nn_pre, "%d ", (int32_t)acc[j]);
	}
#endif
	*pp_output = p_output;
	*pp_kernel = p_kernel;
	*pp_kernel_rec = p_kernel_rec;
	*pp_bias = p_bias;

	return 0;
}

int	fc_8x16(
	int16_t* p_output,
	int8_t* p_kernel,
	int8_t* p_kernel_rec,
	int16_t* p_bias,
	int16_t* input,
	int16_t*input_rec,
	int32_t*c_state,
	int16_t dim_output,
	int16_t dim_input,
	int16_t dim_input_rec,
	int16_t qbit_kernel,
	int16_t qbit_bias,
	int16_t qbit_input, 
	int16_t qbit_input_rec,
	ACTIVATION_TYPE act_type,
	void* (*act)(void*, int32_t*, int))
{
	int16_t* po = p_output;
	int8_t* pw = p_kernel;
	int16_t* pb = p_bias;
	int32_t* po32;

	int i, j;
	int rem_rows = dim_output % 4;
	int groups_4 = dim_output >> 2;
	
	int8_t is_out = 1;
	for (i = 0; i < groups_4; i++)
	{
		
		for (j = 0; j < 4; j++)
			accumulators[j] = 0;
		
		affine_Krows_8x16(4, &po, &pw, &pb, input, dim_input,
							qbit_kernel,
							qbit_bias,
							qbit_input,
							accumulators,
							is_out,
							act);
#if DEBUG_PRINT
		for (j = 0; j < 4; j++)
			fprintf(file_nn_pre, "%d ", (int32_t)acc[j]);
#endif
	}
	if (rem_rows)
	{
		
		for (j = 0; j < rem_rows; j++)
			accumulators[j] = 0;
		
		affine_Krows_8x16(rem_rows, &po, &pw, &pb, input, dim_input,
			qbit_kernel,
			qbit_bias,
			qbit_input,
			accumulators,
			is_out, act);
#if DEBUG_PRINT
		for (j = 0; j < rem_rows; j++)
			fprintf(file_nn_pre, "%d ", (int32_t)acc[j]);
		fprintf(file_nn_pre, "\n");
#endif
	}
	
#if DEBUG_PRINT
	if (act_type != none)
	{
		po = pout;
		for (j = 0; j < rows; j++)
			fprintf(file_nn_out, "%d ", (int16_t) po[j]);
		fprintf(file_nn_out, "\n");
	}
	else
	{
		po32 = (int32_t*) pout;
		for (j = 0; j < rows; j++)
			fprintf(file_nn_out, "%d ", (int32_t) po32[j]);
		fprintf(file_nn_out, "\n");
	}
#endif
	return 0;
}

int rc_8x16(   int16_t* p_output,
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
				void* (*act)(void*, int32_t*, int))
{
	int16_t* po = p_output;
	int8_t* pw = p_kernel;
	int8_t* pw_r = p_kernel_rec;
	int16_t* pb = p_bias;
	int16_t* pb0 = (int16_t*) 0;
	int i, j;
	int groups_4 = dim_output >> 2;
	int rem_rows = dim_output % 4;
	int32_t* po32;
	for (i = 0; i < groups_4; i++)
	{
		rc_Krows_8x16(	
			4, 
			&po,
			&pw, &pw_r, 
			&pb,
			input, input_rec,
			dim_input, dim_input_rec,
			qbit_kernel,
			qbit_bias,
			qbit_input,
			qbit_input_rec,
			act);
	}
	if (rem_rows)
	{
		rc_Krows_8x16(
			rem_rows,
			&po,
			&pw, &pw_r,
			&pb,
			input, input_rec,
			dim_input, dim_input_rec,
			qbit_kernel,
			qbit_bias,
			qbit_input,
			qbit_input_rec,
			act);
	}

#if DEBUG_PRINT
	if (act_type != none)
	{
		po = pout;
		for (j = 0; j < rows; j++)
			fprintf(file_nn_out, "%d ", (int16_t)po[j]);
		fprintf(file_nn_out, "\n");
	}
	else 
	{
		po32 = (int32_t*)pout;
		for (j = 0; j < rows; j++)
			fprintf(file_nn_out, "%d ", (int32_t)po32[j]);
		fprintf(file_nn_out, "\n");
	}
#endif
	return 0;
}


