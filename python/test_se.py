"""
Test trained NN model using wavefile as input
"""
import argparse
import wave
import numpy as np
import soundfile as sf
import sounddevice as sd
from nnsp_pack.feature_module import display_stft_tfmask
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer import NNInferClass
from data_se import params_audio as PARAM_AUDIO

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

class SeClass(NNInferClass):
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

        self.fbank_mel = np.load('fbank_mel.npy')

    def reset(self):
        """
        Reset s2i instance
        """
        super().reset()

    def blk_proc(self, data):
        """
        NN process for several frames
        """
        params_audio = self.params_audio
        file = wave.open(r"output.wav", "wb")
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(16000)

        bks = int(len(data) / params_audio['hop'])
        feats   = []
        specs   = []
        tfmasks =[]
        triggers = data.copy()

        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]
            if NP_INFERENCE:
                feat, spec, est = self.frame_proc_np(data_frame, return_all = True)
            else:
                feat, spec, est = self.frame_proc_tf(data_frame, return_all = True)
            tfmasks += [est]
            feats   += [feat]
            specs   += [spec]
            self.count_run = (self.count_run + 1) % self.num_dnsampl
            print(f"\rprocessing frame {i}", end='')

        print('\n', end='')
        tfmasks = np.array(tfmasks)
        feats   = np.array(feats)
        specs   = np.array(specs)
        display_stft_tfmask(
            data,
            specs.T,
            feats.T,
            tfmasks.T,
            sample_rate=16000)

        out = np.empty((data.size + triggers.size,), dtype=data.dtype)

        out = np.floor(out * 2**15).astype(np.int16)
        file.writeframes(out.tobytes())
        file.close()

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

    se_inst = SeClass(
                    args.nn_arch,
                    epoch_loaded,
                    PARAM_AUDIO,
                    quantized,
                    show_histogram  = SHOW_HISTOGRAM,
                    np_inference    = NP_INFERENCE
                    )

    se_inst.blk_proc(data)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Testing trained SE model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_se_nn_arch.txt',
        help='nn architecture')

    argparser.add_argument(
        '-r',
        '--recording',
        default = 0,
        help    = '1: recording the speech and test it, \
                   0: No recording.')

    argparser.add_argument(
        '-v',
        '--test_wavefile',
        default = 'test_wavs/speech_3.wav',
        help    = 'The wavfile name to be tested')

    argparser.add_argument(
        '-q',
        '--quantized',
        default = True,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default= 8,
        help='starting epoch')

    main(argparser.parse_args())