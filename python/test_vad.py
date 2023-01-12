"""
Test trained NN model using wavefile as input
"""
import os
import argparse
import wave
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import sounddevice as sd
from nnsp_pack.nn_activation import softmax # pylint: disable=no-name-in-module
from nnsp_pack.feature_module import display_stft
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer import NNInferClass
from data_vad import params_audio as PARAM_AUDIO

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

class VadClass(NNInferClass):
    """
    Class to handle VAD model
    """
    def __init__(
            self,
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized=False,
            show_histogram=False,
            np_inference=False):

        super().__init__(
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized,
            show_histogram,
            np_inference)

        self.cnt_vad_trigger = np.zeros(2, dtype=np.int32)
        self.vad_prob = 0.0
        self.vad_trigger = 0

    def reset(self):
        """
        Reset s2i instance
        """
        super().reset()
        self.vad_trigger = 0
        self.vad_prob = 0.0
        self.cnt_vad_trigger *= 0

    def post_nn_infer(self, nn_output):
        """
        post nn inference
        """
        self.vad_trigger  = np.argmax(nn_output)
        self.vad_prob = softmax(nn_output)[1]
        if self.vad_trigger == 0:
            self.cnt_vad_trigger *= 0
        else:
            if self.cnt_vad_trigger[self.vad_trigger] == 0:
                self.cnt_vad_trigger *= 0
            self.cnt_vad_trigger[self.vad_trigger] += 1

    def blk_proc(self, data):
        """
        NN process for several frames
        """
        os.makedirs("test_results", exist_ok=True)
        params_audio = self.params_audio
        file = wave.open(r"test_results/output.wav", "wb")
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(16000)

        bks = int(len(data) / params_audio['hop'])
        feats = []
        specs = []
        triggers = data.copy()
        probs = data.copy()

        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]
            if NP_INFERENCE:
                feat, spec = self.frame_proc_np(data_frame)
            else:
                feat, spec = self.frame_proc_tf(data_frame)

            probs[i*params_audio['hop'] : (i+1) * params_audio['hop']] = self.vad_prob

            if self.cnt_vad_trigger[self.vad_trigger] > 3:
                print(f'\nFrame {i}: trigger')
                triggers[i*params_audio['hop'] : (i+1) * params_audio['hop']] = 0.5
            else:
                print(f'\rFrame {i}:', end='')
                triggers[i*params_audio['hop'] : (i+1) * params_audio['hop']] = 0

            feats += [feat]
            specs += [spec]
            self.count_run = (self.count_run + 1) % self.num_dnsampl

        feats = np.array(feats)
        specs = np.array(specs)

        out = np.empty((data.size + triggers.size,), dtype=data.dtype)
        out[0::2] = data
        out[1::2] = 0.5 * (probs > 0.5).astype(int)
        out = np.floor(out * 2**15).astype(np.int16)
        file.writeframes(out.tobytes())
        file.close()

        display_stft(data, specs.T, feats.T, sample_rate=16000)
        plt.figure(2)
        ax_handle = plt.subplot(3,1,1)
        ax_handle.plot(data)
        ax_handle = plt.subplot(3,1,2)
        ax_handle.plot(probs)
        ax_handle = plt.subplot(3,1,3)
        ax_handle.plot(triggers)
        plt.show()

def main(args):
    """main function"""
    epoch_loaded    = int(args.epoch_loaded)
    quantized       = args.quantized
    recording       = int(args.recording)
    test_wavefile   = args.test_wavefile

    if recording == 1:
        wavefile='test_wavs/speech.wav'
        AudioShowClass(
                record_seconds=10,
                wave_output_filename=wavefile,
                non_stop=False)
    else:
        wavefile = test_wavefile

    data, sample_rate = sf.read(wavefile)

    sd.play(data, sample_rate)

    vad_inst = VadClass(
                    args.nn_arch,
                    epoch_loaded,
                    PARAM_AUDIO,
                    quantized,
                    show_histogram  = SHOW_HISTOGRAM,
                    np_inference    = NP_INFERENCE
                    )

    vad_inst.blk_proc(data)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Testing trained VAD model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_vad_nn_arch.txt',
        help='nn architecture')

    argparser.add_argument(
        '-r',
        '--recording',
        default = 1,
        help    = '1: recording the speech and test it, \
                   0: No recording.')

    argparser.add_argument(
        '-v',
        '--test_wavefile',
        default = 'test_wavs/speech.wav',
        help    = 'The wavfile name to be tested')

    argparser.add_argument(
        '-q',
        '--quantized',
        default = True,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default= 811,
        help='starting epoch')

    main(argparser.parse_args())
