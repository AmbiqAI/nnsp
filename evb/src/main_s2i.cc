#include <stdint.h>
#include "s2iCntrlClass.h"
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_audio.h"
#include "ambiq_nnsp_const.h"
#include "arm_intrinsic_test.h"
#include "ns_timer.h"
#include <cmsis_gcc.h>
#define NUM_CHANNELS 1
int volatile g_intButtonPressed = 0;
///Button Peripheral Config Struct
ns_button_config_t button_config_nnsp = {
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &g_intButtonPressed,
    .button_1_flag = NULL
};
/// Set by app when it wants to start recording, used by callback
bool static g_audioRecording = false;
/// Set by callback when audio buffer has been copied, cleared by
/// app when the buffer has been consumed.
bool static g_audioReady = false;
/// Audio buffer for application
int16_t static g_in16AudioDataBuffer[LEN_STFT_HOP * 2];
/**
* 
* @brief Audio Callback (user-defined, executes in IRQ context)
* 
* When the 'g_audioRecording' flag is set, copy the latest sample to a buffer
* and set a 'ready' flag. If recording flag isn't set, discard buffer.
* If 'ready' flag is still set, the last buffer hasn't been consumed yet,
* print a debug message and overwrite.
* 
*/
void
audio_frame_callback(ns_audio_config_t *config, uint16_t bytesCollected) {
    uint32_t *pui32_buffer =
        (uint32_t *)am_hal_audadc_dma_get_buffer(config->audioSystemHandle);

    if (g_audioRecording) {
        if (g_audioReady) {
            ns_printf("Warning - audio buffer wasnt consumed in time\n");
        }
        // Raw PCM data is 32b (14b/channel) - here we only care about one
        // channel For ringbuffer mode, this loop may feel extraneous, but it is
        // needed because ringbuffers are treated a blocks, so there is no way
        // to convert 32b->16b
        for (int i = 0; i < config->numSamples; i++) {
            g_in16AudioDataBuffer[i] = (int16_t)(pui32_buffer[i] & 0x0000FFF0);
        }
#ifdef RINGBUFFER_MODE
        ns_ring_buffer_push(&(config->bufferHandle[0]),
                                      g_in16AudioDataBuffer,
                                      (config->numSamples * 2), // in bytes
                                      false);
#endif
        g_audioReady = true;
    }
}

/**
 * @brief NeuralSPOT Audio config struct
 * 
 * Populate this struct before calling ns_audio_config()
 * 
 */
ns_audio_config_t audio_config = {
#ifdef RINGBUFFER_MODE
    .eAudioApiMode = NS_AUDIO_API_RINGBUFFER,
    .callback = audio_frame_callback,
    .audioBuffer = (void *)&pui8AudioBuff,
#else
    .eAudioApiMode = NS_AUDIO_API_CALLBACK,
    .callback = audio_frame_callback,
    .audioBuffer = (void *)&g_in16AudioDataBuffer,
#endif
    .eAudioSource = NS_AUDIO_SOURCE_AUDADC,
    .numChannels = NUM_CHANNELS,
    .numSamples = LEN_STFT_HOP,
    .sampleRate = SAMPLING_RATE,
    .audioSystemHandle = NULL, // filled in by audio_init()
#ifdef RINGBUFFER_MODE
    .bufferHandle = audioBuf
#else
    .bufferHandle = NULL
#endif
};

int main(void) {

    int64_t val64 = 0;
    int32_t val32 = 122, val32_1 = ~(((int32_t) 1) << 31);
    int8_t val8 = 122;
    int i,j;
    uint32_t elapsed_time;
    s2iCntrlClass cntrl_inst;
    g_audioRecording = false;
    ns_itm_printf_enable();
    
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
    //
    // Initialize the printf interface for ITM output
    //
    ns_debug_printf_enable();
    ns_power_config(&ns_development_default);
    ns_audio_init(&audio_config);
    ns_peripheral_button_init(&button_config_nnsp);

    // initialize neural nets controller
    s2iCntrlClass_init(&cntrl_inst);
    
    // reset all internal states
    s2iCntrlClass_reset(&cntrl_inst);
    ns_printf("\nPress button to start!\n");
    
    
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

    ns_timer_init(0);
    for (i = 0; i < 1000; i++)
    {
        s2iCntrlClass_exec(&cntrl_inst, g_in16AudioDataBuffer);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_printf("%3.5f ms/inference\n", ((float) elapsed_time) / 1000 / 1000) ;
    
    // reset all internal states
    s2iCntrlClass_reset(&cntrl_inst);
    while (1) 
    {
        g_audioRecording = false;
        g_intButtonPressed = 0;
        
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
        
        if ( (g_intButtonPressed == 1) && (!g_audioRecording) ) 
        {
            ns_printf("\nYou'd pressed the button. Program start!\n");
            g_intButtonPressed = 0;
            g_audioRecording = true;
            am_hal_delay_us(10);   
            while (1)
            {   
                am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);

                if (g_audioReady) 
                {
                    // execution of each time frame data
                    s2iCntrlClass_exec(&cntrl_inst, g_in16AudioDataBuffer);
                    g_audioReady = false;
                }
            }
            ns_printf("\nPress button to start!\n");
        }
    } // while(1)
}
