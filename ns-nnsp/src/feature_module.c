#include "ambiq_stdint.h"
#include "spectrogram_module.h"
#include "feature_module.h"
#include "minmax.h"
#include "ambiq_nnsp_const.h"
#include "melSpecProc.h"
#include "fixlog10.h"
#define LOG10_2POW_N15_Q15 (-147963)
int32_t spec[LEN_FFT_NNSP+2];

void FeatureClass_construct(
	FeatureClass* ps,
		const int32_t* norm_mean,
		const int32_t* norm_stdR,
		int8_t qbit_output)
{
	stftModule_construct(&ps->state_stftModule);
	ps->pt_norm_mean = norm_mean;
	ps->pt_norm_stdR = norm_stdR;
	ps->num_context = NUM_FEATURE_CONTEXT;
	ps->dim_feat = DIMEMSION_FEATURE;
	ps->qbit_output = qbit_output;
}

void FeatureClass_setDefault(FeatureClass* ps)
{
	int i, j;
	int64_t tmp64;
	int16_t tmp;
	stftModule_setDefault(&ps->state_stftModule);
	for (i = 0; i < ps->dim_feat; i++)
	{
		tmp64 = (int64_t)((int32_t)LOG10_2POW_N15_Q15 - ps->pt_norm_mean[i]);
		tmp64 = (tmp64 * (int64_t)ps->pt_norm_stdR[i]) >> (30 - ps->qbit_output);
		tmp64 = MIN(MAX(tmp64, (int64_t)MIN_INT16_T), (int64_t)MAX_INT16_T);
		tmp = (int16_t) tmp64;

		for (j = 0; j < (ps->num_context - 1); j++)
		{
			ps->normFeatContext[i + j * ps->dim_feat] = tmp;
		}

	}
}

void FeatureClass_execute(FeatureClass*ps,
							int16_t* input)
{
	int32_t* pspec = spec;
	int shift = (ps->num_context - 1) * ps->dim_feat;
	int i;
	int64_t tmp;
	for (i = 0; i < shift; i++)
	{
		ps->normFeatContext[i] = ps->normFeatContext[i + ps->dim_feat];
	}
	stftModule_analyze(&ps->state_stftModule, input, spec);
	spec2pspec(pspec, spec, 1 + (LEN_FFT_NNSP >> 1));
	melSpecProc(pspec, ps->feature);
	log10_vec(ps->feature, ps->feature, ps->dim_feat, 15);
	for (i = 0; i < ps->dim_feat; i++)
	{
		tmp = (int64_t) ps->feature[i] - (int64_t) ps->pt_norm_mean[i];
		tmp = (tmp * ((int64_t)ps->pt_norm_stdR[i])) >> (30 - ps->qbit_output); //Bit_frac_out = 30-22 = 8
		tmp = MIN(MAX(tmp, (int64_t) MIN_INT16_T), (int64_t) MAX_INT16_T);
		ps->normFeatContext[i + shift] = (int16_t) tmp;
	}

}