#include <stdint.h>
#include <cmsis_gcc.h>
#include "s2iCntrlClass.h"
// #include "nnCntrlClass.h"
#include "arm_intrinsic_test.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_timer.h"

int arm_test_s2i(
        void *pt_cntrl_inst_, 
        int16_t *pt_data)
{
    #if 0
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val32 = __SMLAD(val32, val32, val32);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%u\n", elapsed_time);

    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 = __SMLALD(val32, val32, val64);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%u\n", elapsed_time);
    
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 += (int64_t) val8 * (int64_t) val8 + (int64_t) val8 * (int64_t) val8; 
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%u\n", elapsed_time);
    #endif
    int i;
    uint32_t elapsed_time;
    s2iCntrlClass *pt_cntrl_inst = (s2iCntrlClass *) pt_cntrl_inst_; 
    // reset all internal states
    s2iCntrlClass_reset(pt_cntrl_inst);
    ns_timer_init(0);
    for (i = 0; i < 1000; i++)
    {
        s2iCntrlClass_exec(pt_cntrl_inst, pt_data);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%3.5f ms/inference\n", ((float) elapsed_time) / 1000 / 1000) ;
    
    

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
    ns_printf("%u\n", elapsed_time);

    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 = __SMLALD(val32, val32, val64);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%u\n", elapsed_time);
    
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 += (int64_t) val8 * (int64_t) val8 + (int64_t) val8 * (int64_t) val8; 
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%u\n", elapsed_time);
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
    ns_printf("%3.5f ms/inference\n", ((float) elapsed_time) / 1000 / 1000) ;
    
    

    return 0;
}
#endif