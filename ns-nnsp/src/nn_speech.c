#include <stdint.h>
#include "ambiq_stdint.h"
#include "s2i_const.h"
#include "ambiq_nnsp_const.h"
#include "nn_speech.h"
#include "feature_module.h"
#include "neural_nets.h"
#include "minmax.h"
#include "nnsp_identification.h"
#define CHECK_POWER 0

#if CHECK_POWER
#include "timer.h"
#include "am_hal_pwrctrl.h"
#include "apollo4b.h"
#endif

#if DEBUG_PRINT
#include <math.h>
#include <stdio.h>
#endif

int NNSPClass_init(
        NNSPClass* pt_inst,
        void* pt_net,
        void *pt_feat,
        char nn_id,
        const int32_t* pt_mean,
        const int32_t* pt_stdR,
	    int16_t * pt_thresh_prob,
	    int16_t *pt_th_count_trigger)
{

    pt_inst->nn_id = nn_id;
    
    pt_inst->pt_feat = (void*) pt_feat;

    pt_inst->pt_net  = (void*) pt_net;

    FeatureClass_construct(
            (FeatureClass*)pt_inst->pt_feat,
            pt_mean,
            pt_stdR,
            ((NeuralNetClass*) pt_inst->pt_net)->qbit_input[0]);

    pt_inst->num_dnsmpl = 2;

    pt_inst->pt_thresh_prob = pt_thresh_prob;

    pt_inst->pt_th_count_trigger = pt_th_count_trigger;

    NeuralNetClass_init((NeuralNetClass*) pt_inst->pt_net);
    
	return 0;
}

int NNSPClass_reset(NNSPClass* pt_inst)
{
	int i;
    FeatureClass_setDefault((FeatureClass*)   pt_inst->pt_feat);
    NeuralNetClass_setDefault((NeuralNetClass*) pt_inst->pt_net);
    pt_inst->slides = 1;
    pt_inst->trigger = 0;
    for (i = 0; i < DIM_INTENTS; i++)
        pt_inst->counts_category[i] = 0;

    for (i = 0; i < 3; i++)
        pt_inst->outputs[i] = 0;

    pt_inst->argmax_last = 0;
	return 0;
}

int16_t NNSPClass_exec(
        NNSPClass* pt_inst,
        int16_t* rawPCM)
{
	static int32_t output[50];
    FeatureClass* pt_feat = (FeatureClass*)  pt_inst->pt_feat;
    NeuralNetClass* pt_net = (NeuralNetClass*) pt_inst->pt_net;
    int8_t debug_layer = -1;
    FeatureClass_execute(pt_feat, rawPCM);
    
    if (pt_inst->slides == 1)
    {
        #ifdef ENERGYMODE
            am_set_power_monitor_state(AM_AI_INFERING);
        #endif
#if CHECK_POWER
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);
#endif
        NeuralNetClass_exe(
            pt_net, 
            pt_feat->normFeatContext, 
            output,
            debug_layer);
        switch (pt_inst->nn_id)
        {
        case s2i_id:
            s2i_post_proc(
                pt_inst,
                output,
                &pt_inst->trigger);
            break;

        case kws_galaxy_id:
            binary_post_proc(
                pt_inst,
                output,
                &pt_inst->trigger);
            break;

        case vad_id:
            binary_post_proc(
                pt_inst,
                output,
                &pt_inst->trigger);
            break;
        }

#if CHECK_POWER
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_LOW_POWER);
#endif
    }
    pt_inst->slides = (pt_inst->slides + 1) % 2;
	return pt_inst->trigger;
}


void my_argmax(int32_t *vals, int len, int16_t *pt_argmax)
{
    int i;
    int32_t max_val;
    *pt_argmax = 0;
    max_val = vals[0];
    for (i = 1; i < len; i++)
    {
        if (vals[i] >= max_val)
        {
            max_val = vals[i];
            *pt_argmax = i;
        }
    }
}

void s2i_post_proc(
        NNSPClass* pt_inst,
        int32_t* pt_nn_est,
        int16_t*pt_trigger)
{
    int i;
    int32_t *intent = pt_nn_est;
    int32_t *slot0  = pt_nn_est + DIM_INTENTS;
    int32_t *slot1  = slot0 + DIM_SLOTS;
    int16_t argmax_intent, argmax;  
  
    *pt_trigger=0;
    for (i = 0; i < 3; i++)
        pt_inst->outputs[i] = 0;

    my_argmax(intent, DIM_INTENTS, &argmax_intent);
    
    if ((pt_inst->argmax_last == 0 ) || (pt_inst->argmax_last == argmax_intent))
    {
        if (argmax_intent != 0 )
        {
            pt_inst->counts_category[argmax_intent]++;
            if (pt_inst->counts_category[argmax_intent] > *pt_inst->pt_th_count_trigger)
            {
                *pt_trigger = 1;
                pt_inst->outputs[0] = argmax_intent;

                my_argmax(slot0, DIM_SLOTS, &argmax);
                pt_inst->outputs[1] = argmax;
                
                my_argmax(slot1, DIM_SLOTS, &argmax);
                pt_inst->outputs[2] = argmax;

            }
        }
    }
    else
    {
        for (i = 0; i < DIM_INTENTS; i++)
            pt_inst->counts_category[i] = 0;
    }

    pt_inst->argmax_last = argmax_intent;
}

void binary_post_proc(
        NNSPClass* pt_inst,
        int32_t* pt_est,
        int16_t *pt_trigger)
{
    int32_t max_val;
    int32_t i;
    static int32_t log2_e = (int32_t) 0x0000B8AA;
    int32_t thresh;
    int32_t tmp;
    int32_t val;
    int32_t den;
    int64_t ref;
    int16_t pre_trigger;

    max_val = MAX(pt_est[0], pt_est[1]);
  
    for (i = 0; i < 2; i++)
    {
        val = pt_est[i] - max_val;
        ref = ((int64_t) val * (int64_t)log2_e) >> 15;
        ref = MIN(MAX(ref, MIN_INT32_T), MAX_INT32_T);
        pt_est[i] = compute_pwr2((int32_t) ref);
    }
    den = pt_est[0] + pt_est[1];

    thresh = ONE_N32_Q15 - (int) *pt_inst->pt_thresh_prob;
    ref = ((int64_t) thresh * (int64_t)den) >> 15;
    tmp = (int32_t) ref;
    pre_trigger = (pt_est[0] <= tmp) ? 1 : 0;
    if (pre_trigger)
        pt_inst->counts_category[0] += 1;
    else
        pt_inst->counts_category[0] = 0;

    *pt_trigger = (pt_inst->counts_category[0] >= *pt_inst->pt_th_count_trigger) ? 1 : 0;
}

int32_t ceiling(int32_t input)
{
    int32_t output = (input >> 15);
    output <<= 15;
    output = ((output - input) == 0) ? output : output + ONE_N32_Q15;
    return output;
}

int32_t compute_pwr2(int32_t input)
{
    static int16_t coeff_pwr[] = { 0x5a82, 0x1fd7, 0x057a };
    int32_t t;
    int32_t output;
    int32_t input_ceil = ceiling(input);
    int32_t shift;

    input = input - input_ceil;
    shift = input_ceil >> 15;

    if (shift <= -15)
        output = 0;
    else
    {
        t = (input << 1) + ONE_N32_Q15;
        output = (int32_t) coeff_pwr[1] + ((t * (int32_t) coeff_pwr[2]) >> 15);
        output = (int32_t) coeff_pwr[0] + ((t * output) >> 15);
        output = (shift < 0) ? output >> -shift : output << shift;
    }
    return output;
}
