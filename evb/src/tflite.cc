#include "arm_math.h"
#include <stdio.h>
#include "ns_ambiqsuite_harness.h"
#include "tflite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "quant_model_act.h"

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* pt_model = nullptr;
tflite::MicroInterpreter* pt_interpreter = nullptr;
TfLiteTensor* pt_input = nullptr;
TfLiteTensor* pt_output = nullptr;
constexpr int kTensorArenaSize = 200000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Model Stuff

void tflite_init(void)
{
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    tflite::InitializeTarget();

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    pt_model = tflite::GetModel(model_tflite);
    if (pt_model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            pt_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // This pulls in all the operation implementations we need.

    //static tflite::MicroMutableOpResolver<1> resolver;
    static tflite::AllOpsResolver resolver;
    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        pt_model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        error_reporter);
    
    pt_interpreter = &static_interpreter;
    ns_lp_printf("Initialization\n");
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = pt_interpreter->AllocateTensors();
    ns_lp_printf("allocate_status = %d\n", allocate_status);
    if (allocate_status != kTfLiteOk) 
    {
        ns_lp_printf("Initialization tflite Failed\n");
        return;
    }
    else
        ns_lp_printf("Initialization tflite ok\n");
    
    // Obtain pointers to the model's input and output tensors.
    pt_input = pt_interpreter->input(0);
    pt_output= pt_interpreter->output(0);

    ns_lp_printf("pt_input->dims->size %d\n",pt_input->dims->size);
    for (int i = 0; i < pt_input->dims->size; i++)
        ns_lp_printf("pt_input->dims->data[0] %d\n", pt_input->dims->data[i]);
    ns_lp_printf("type pt_input->type %d\n", pt_input->type);
    ns_lp_printf("\n");

    ns_lp_printf("pt_output->dims->size %d\n", pt_output->dims->size);
    for (int i = 0; i < pt_output->dims->size; i++)
        ns_lp_printf("pt_output->dims->data[0] %d\n", pt_output->dims->data[i]);
    ns_lp_printf("pt_output->type %d\n", pt_output->type);
    ns_lp_printf("\n");
}

int test_tflite(void)
{
    for (int fr = 0; fr < 2; fr++)
    {
        for (int j =0; j < 100; j++)
            pt_input->data.i16[j] = j;
        TfLiteStatus invoke_status = pt_interpreter->Invoke();
        if (invoke_status != kTfLiteOk)
            ns_lp_printf("Invoke failed");
        
        ns_lp_printf("Frame %d: %d\n", fr, pt_output->data.i16[1]);
    }

    return 0;
}