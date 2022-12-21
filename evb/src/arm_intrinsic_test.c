#include <stdint.h>
#include <cmsis_gcc.h>
#include "s2iCntrlClass.h"
// #include "nnCntrlClass.h"
#include "arm_intrinsic_test.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_timer.h"
#include "fft_arm.h"
#include "feature_module.h"
#include "neural_nets.h"
// #include "def_nn3_se.h"
#define NUM_FRAMES_EST 1000

ns_timer_config_t my_tickTimer = {
    .prefix = {0},
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};
#include "ns_rpc_generic_data.h"
int16_t static gin16AudioDataBuffer[LEN_STFT_HOP << 1];
static char msg_store[30] = "Audio16bPCM_to_WAV";
// Block sent to PC
static dataBlock outBlock = {
    .length = LEN_STFT_HOP * sizeof(int16_t),
    .dType = uint8_e,
    .description = msg_store,
    .cmd = write_cmd,
    .buffer = {.data = (uint8_t *)gin16AudioDataBuffer, // point this to audio buffer
               .dataLength = LEN_STFT_HOP * sizeof(int16_t)}};

static ns_rpc_config_t rpcConfig = {.mode = NS_RPC_GENERICDATA_CLIENT,
                                    .sendBlockToEVB_cb = NULL,
                                    .fetchBlockFromEVB_cb = NULL,
                                    .computeOnEVB_cb = NULL};
// int arm_test_se()
// {
//     extern NeuralNetClass net_se;
//     int16_t input[257];
//     int32_t output[257];
//     uint32_t elapsed_time;
//     NeuralNetClass_init(&net_se);
//     NeuralNetClass_setDefault(&net_se);
//     ns_timer_init(&my_tickTimer);
//     for (int i = 0; i < NUM_FRAMES_EST; i++)
//     {
//         NeuralNetClass_exe(&net_se, input, output, -1);
//     }
//     elapsed_time = ns_us_ticker_read(&my_tickTimer);    
//     ns_lp_printf("Total: %3.2f ms/inference (se)\n",
//                 ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
//     return 0;
// }

int arm_test_s2i(
        void *pt_cntrl_inst_, 
        int16_t *pt_data)
{
    uint32_t elapsed_time;
    s2iCntrlClass *pt_cntrl_inst = (s2iCntrlClass *) pt_cntrl_inst_; 
    
    // reset all internal states
    s2iCntrlClass_reset(pt_cntrl_inst);
	ns_timer_init(&my_tickTimer);
    for (int i = 0; i < NUM_FRAMES_EST; i++)
    {
        s2iCntrlClass_exec(pt_cntrl_inst, pt_data);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
    ns_lp_printf("Total: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}

uint32_t test_feat()
{
    FeatureClass feat;
    uint32_t elapsed_time;
    int32_t mean[50];
    int32_t stdR[50];
    int16_t input[160];
    int i;

    FeatureClass_construct(&feat, mean, stdR, 15);
    FeatureClass_setDefault(&feat);
	ns_timer_init(&my_tickTimer);
    for (i=0; i < NUM_FRAMES_EST; i++)
    {
        FeatureClass_execute(&feat, input);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
    ns_lp_printf("feat: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}

uint32_t test_fft()
{
    int32_t input[512 + 2];
    int32_t output[512 << 1];
    uint32_t elapsed_time;
    int i;
    arm_fft_init();
	ns_timer_init(&my_tickTimer);
    for (i = 0; i < NUM_FRAMES_EST; i++)
    {
        arm_fft_exec(output, input);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
    ns_lp_printf("fft: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}
uint32_t test_upload_pc()
{
    int32_t input[512 + 2];
    int32_t output[512 << 1];
    uint32_t elapsed_time;
    int i;
    arm_fft_init();
	ns_timer_init(&my_tickTimer);
    ns_rpc_genericDataOperations_init(&rpcConfig); // init RPC and USB
    ns_lp_printf("\nStart the PC-side server...\n");
    for (i = 0; i < NUM_FRAMES_EST; i++)
    {
        ns_rpc_data_sendBlockToPC(&outBlock);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
    ns_lp_printf("rpc_client: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}
#if 0
int arm_test_nnsp(
        void *pt_cntrl_inst_, 
        int16_t *pt_data,
        int16_t *data_buf)
{
    #if 0
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val32 = __SMLAD(val32, val32, val32);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%u\n", elapsed_time);

    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 = __SMLALD(val32, val32, val64);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%u\n", elapsed_time);
    
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 += (int64_t) val8 * (int64_t) val8 + (int64_t) val8 * (int64_t) val8; 
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%u\n", elapsed_time);
    #endif
    nnCntrlClass *pt_cntrl_inst = (nnCntrlClass *) pt_cntrl_inst_;
    int i;
    uint32_t elapsed_time;
    // reset all internal states
    nnCntrlClass_reset(pt_cntrl_inst);
    ns_timer_init(0);
    for (i = 0; i < 1000; i++)
    {
        nnCntrlClass_exec(pt_cntrl_inst, pt_data, data_buf);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%3.5f ms/inference\n", ((float) elapsed_time) / 1000 / 1000) ;
    
    

    return 0;
}
#endif