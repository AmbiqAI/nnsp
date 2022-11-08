"""
Test trained NN model using wavefile as input
"""
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd
from nnsp_pack.feature_module import display_stft
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer import NNInferClass
from train_s2i import DIM_INTENT, DIM_SLOT
from data_s2i import intent_ids, slot_ids
from data_s2i import params_audio as PARAMS_AUDIO

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

class S2iClass(NNInferClass):
    """
    Class to handle s2i model
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

        self.ids_intent = {}
        for key, _ in intent_ids.items():
            self.ids_intent[intent_ids[key]] = key

        self.ids_slot = {}
        for key, _ in slot_ids.items():
            self.ids_slot[slot_ids[key]] = key

        self.cnt_intents = np.zeros(DIM_INTENT, dtype=np.int32)
        self.intent = 0
        self.slot0  = 0
        self.slot1  = 0

    def reset(self):
        """
        Reset s2i instance
        """
        super().reset()
        self.cnt_intents *= 0
        self.intent = 0
        self.slot0  = 0
        self.slot1  = 0

    def post_nn_infer(self, nn_output):
        """
        post nn inference
        """
        self.intent  = np.argmax(nn_output[:DIM_INTENT])
        self.slot0   = np.argmax(nn_output[DIM_INTENT:DIM_SLOT + DIM_INTENT])
        self.slot1   = np.argmax(nn_output[DIM_SLOT + DIM_INTENT:])
        if self.intent == 0:
            self.cnt_intents *= 0
        else:
            if self.cnt_intents[self.intent] == 0:
                self.cnt_intents *= 0
            self.cnt_intents[self.intent] += 1

    def blk_proc(self, data):
        """
        NN process for several frames
        """
        params_audio = self.params_audio
        bks = int(len(data) / params_audio['hop'])
        feats = []
        specs = []
        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]
            if NP_INFERENCE:
                feat, spec = self.frame_proc_np(data_frame)
            else:
                feat, spec = self.frame_proc_tf(data_frame)

            if self.cnt_intents[self.intent] > 5:
                print(f'\nFrame {i}: {self.ids_intent[self.intent]},',
                                        f'{ self.ids_slot[self.slot0]},',
                                        f'{self.ids_slot[self.slot1]}')
                self.reset()
            else:
                print(f'\rFrame {i}:', end='')

            feats += [feat]
            specs += [spec]

            self.count_run = (self.count_run + 1) % self.num_dnsampl

        feats = np.array(feats)
        specs = np.array(specs)
        display_stft(data, specs.T, feats.T, sample_rate=16000)

def main(args):
    """main function"""
    epoch_loaded    = int(args.epoch_loaded)
    quantized       = args.quantized
    recording       = int(args.recording)
    test_wavefile   = args.test_wavefile

    if recording==1:
        wavefile='./test_wavs/speech.wav'
        AudioShowClass(
                record_seconds=10,
                wave_output_filename=wavefile,
                non_stop=False)
    else:
        wavefile = test_wavefile

    data, sample_rate = sf.read(wavefile)

    sd.play(data, sample_rate)

    s2i_inst = S2iClass(
            args.nn_arch,
            epoch_loaded,
            PARAMS_AUDIO,
            quantized,
            show_histogram  = SHOW_HISTOGRAM,
            np_inference    = NP_INFERENCE
            )

    s2i_inst.blk_proc(data)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Testing trained s2i model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_s2i_nn_arch.txt',
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
        default = './test_wavs/speech.wav',
        help    = 'The wavfile name to be tested')

    argparser.add_argument(
        '-q',
        '--quantized',
        default = False,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default= 800,
        help='starting epoch')

    main(argparser.parse_args())
