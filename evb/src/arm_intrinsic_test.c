#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "arm_intrinsic_test.h"
#include "ns_ambiqsuite_harness.h"
#include <stdint.h>

int arm_test()
{
    int64_t sum = 0;
    int8_t arry[4] = {100,2,-3,4};
    int32_t out;
    int16_t *po = (int16_t*) &out;
    int32_t *p_arry = (int32_t*) arry;
    int16_t pt_a[2] = {3,12};
    int16_t pt_b[2] = {8,4};
    int32_t *pa = (int32_t*) pt_a;
    int32_t *pb = (int32_t*) pt_b;
    sum = 	__SMLALD(*pa, *pb, sum);
    ns_printf("sum=%lld\n", sum);
    
    out = __SXTB16(__ROR(*p_arry, 8));
    ns_printf("po[0:1]=[%d,%d]\n", po[0], po[1]);

    return sum;
}

