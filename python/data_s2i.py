"""
1. Synthesize audio data
2. Feature extraction for audio data.
"""
import os
import random
import time
import argparse
import re
import multiprocessing
import logging
import numpy as np
import wandb
import boto3
import soundfile as sf
import sounddevice as sd
from nnsp_pack import tfrecord_converter_s2i
from nnsp_pack.feature_module import FeatureClass, display_stft
from nnsp_pack import add_noise
from nnsp_pack import add_garbage
from nnsp_pack import boto3_op

DEBUG = False
UPLOAD_TFRECORD_S3 = False
DOWLOAD_DATA = False

if UPLOAD_TFRECORD_S3:
    print('uploading tfrecords to s3 will slow down the process')

intent_ids = {
    'none'              : 0,
    'change language'   : 1,
    'bring'             : 2,
    'activate'          : 3,
    'deactivate'        : 4,
    'increase'          : 5,
    'decrease'          : 6 }

slot_ids = {
    'none'              : 0,
    'washroom'          : 1,
    'juice'             : 2,
    'volume'            : 3,
    'shoes'             : 4,
    'music'             : 5,
    'heat'              : 6,
    'lights'            : 7,
    'kitchen'           : 8,
    'newspaper'         : 9,
    'lamp'              : 10,
    'chinese'           : 11,
    'german'            : 12,
    'korean'            : 13,
    'english'           : 14,
    'bedroom'           : 15,
    'socks'             : 16 }

S3_BUCKET = "ambiqai-speech-commands-dataset"
S3_PREFIX = "tfrecords"


params_audio = {
    'win_size'      : 480,
    'hop'           : 160,
    'len_fft'       : 512,
    'sample_rate'   : 16000,
    'nfilters_mel'  : 40
}

def download_data():
    """
    download data
    """
    s3 = boto3.client('s3')
    audio_lists = [
        'data/test_files_s2i.csv',
        'data/train_files_s2i.csv',
        'data/noise_list.csv',
        'data/test_garb.csv',
        'data/train_garb.csv'
    ]
    boto3_op.s3_download(S3_BUCKET, audio_lists)
    return s3

class FeatMultiProcsClass(multiprocessing.Process):
    """
    FeatMultiProcsClass use multiprocesses
    to run several processes of feature extraction in parallel
    """
    def __init__(self, id_process,
                 name, src_list, train_set, ntype,
                 noise_files, snr_db,
                 garb_files, success_dict,
                 params_audio_def):

        multiprocessing.Process.__init__(self)
        self.success_dict = success_dict
        self.id_process         = id_process
        self.name               = name
        self.src_list           = src_list
        self.params_audio_def   = params_audio_def
        self.feat_inst      = FeatureClass(
                                win_size        = params_audio_def['win_size'],
                                hop             = params_audio_def['hop'],
                                len_fft         = params_audio_def['len_fft'],
                                sample_rate     = params_audio_def['sample_rate'],
                                nfilters_mel    = params_audio_def['nfilters_mel'])

        self.train_set      = train_set
        self.ntype          = ntype
        self.noise_files    = noise_files
        self.garb_files     = garb_files
        self.snr_db         = snr_db
        if DEBUG:
            self.cnt = 0

    def run(self):
        #      threadLock.acquire()
        print("Running " + self.name)

        self.convert_tfrecord(
                    self.src_list,
                    self.id_process)

    def convert_tfrecord(
            self,
            fnames,
            id_process):
        """
        convert np array to tfrecord
        """
        width_for_target = 30
        for i, fname in enumerate(fnames):
            success = 1
            sps = fname.strip().split(',')
            wavpath = sps[0]
            intent = np.array(intent_ids[sps[2].lower()], dtype=np.int32)
            slot = np.array([slot_ids[sps[3].lower()], slot_ids[sps[4].lower()]], dtype=np.int32)
            stime = int(sps[5])         # start time
            etime = int(sps[6])         # end time
            tfrecord = re.sub(r'\.wav$', '.tfrecord', re.sub(r'wavs', S3_PREFIX, wavpath))

            try:
                audio, sample_rate = sf.read(wavpath)
            except :# pylint: disable=bare-except
                success = 0
                logging.debug("Reading the %s fails ", wavpath)
            else:
                if audio.ndim > 1:
                    audio=audio[:,0]
                else:
                    pass
                # decorate speech
                speech = audio[stime : etime]
                stime = np.random.randint(sample_rate >> 1, sample_rate << 1)
                zeros_s = np.zeros(stime)

                size_zeros = np.random.randint(sample_rate >> 1, sample_rate << 1)
                zeros_e = np.zeros(size_zeros)
                etime = len(speech) + stime
                speech = np.concatenate((zeros_s, speech, zeros_e))

                # load garbage
                garb = add_garbage.get_garbage_audio(self.garb_files[self.train_set])

                # concatenate sig = speech + garbage
                audio, stime, etime = add_garbage.concat_garb(garb, speech, stime, etime)

                start_frame = int(stime / self.params_audio_def['hop']) + 1 # target level frame
                end_frame   = int(etime / self.params_audio_def['hop']) + 1 # target level frame

                # add noise to sig
                noise = add_noise.get_noise(self.noise_files[self.train_set], len(audio))
                audio = add_noise.add_noise(audio, noise, self.snr_db, stime, etime)
                # feature extraction of sig
                spec, _, feat, _ = self.feat_inst.block_proc(audio)

                if DEBUG:
                    sd.play(audio, sample_rate)
                    print(fname)
                    print(intent)
                    print(slot)
                    print(f'Ending frame = {end_frame}')
                    flabel = np.zeros(spec.shape[0])
                    flabel[start_frame: end_frame] = 1
                    display_stft(audio, spec.T, feat.T, sample_rate, label_frame=flabel)

                    sf.write(f'test_wavs/speech_{self.cnt}.wav', audio, sample_rate)
                    sf.write(f'test_wavs/speech_{self.cnt}_ref.wav', speech, sample_rate)

                    self.cnt = self.cnt + 1

            if success:
                ntype = re.sub('/','_', self.ntype)
                tfrecord = re.sub(  r'\.tfrecord$',
                                    f'_snr{self.snr_db}dB_{ntype}.tfrecord',
                                     tfrecord)
                os.makedirs(os.path.dirname(tfrecord), exist_ok=True)
                try:
                    timesteps, _  = feat.shape
                    tfrecord_converter_s2i.make_tfrecord(
                        tfrecord,
                        feat,
                        intent,
                        slot[0], slot[1],
                        timesteps, end_frame, width_for_target)
                except: # pylint: disable=bare-except
                    logging.debug("Thread-%d: %d, processing %s failed", id_process, i, tfrecord)
                else:
                    # since tfrecord file starts: data/tfrecords/speakers/...
                    # strip the leading "data/" when uploading
                    self.success_dict[self.id_process] += [tfrecord]
                    if UPLOAD_TFRECORD_S3:
                        s3.upload_file(tfrecord, S3_BUCKET, tfrecord)
                    else:
                        pass

def main(args):
    """
    main function to generate all training and testing data
    """
    if DOWLOAD_DATA:
        s3 = download_data()
    if args.wandb_track:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="data-update")
        wandb.config.update(args)

    train_sets = ['train', 'test']

    ntypes = ['musan/noise/sound-bible',
              'musan/noise/free-sound',
              'musan/music']

    if DEBUG:
        snr_dbs = [20]
    else:
        snr_dbs = [5, 10, 20, 100]

    target_files = { 'train': args.train_dataset_path,
                     'test' : args.test_dataset_path}

    garb_files = { 'train': add_garbage.get_garbage_files('data/train_garb.csv'),
                    'test': add_garbage.get_garbage_files('data/test_garb.csv')}

    tot_success_dict = {'train': [], 'test': []}
    for train_set in train_sets:
        with open(target_files[train_set], 'r') as file: # pylint: disable=unspecified-encoding
            filepaths = file.readlines()[1:]

        blk_size = int(np.floor(len(filepaths) / args.num_procs))
        sub_src = []
        for i in range(args.num_procs):
            idx0 = i * blk_size
            if i == args.num_procs - 1:
                sub_src += [filepaths[idx0:]]
            else:
                sub_src += [filepaths[idx0:blk_size+idx0]]

        for snr_db in snr_dbs:
            for ntype in ntypes:
                manager = multiprocessing.Manager()
                success_dict = manager.dict({i: [] for i in range(args.num_procs)})
                print(f'{train_set} set running: snr_dB = {int(snr_db)}, ntype={ntype}')
                lst_ns = add_noise.get_noise_files_new(ntype)
                random.shuffle(lst_ns)
                start = int(len(lst_ns) / 5)
                noise_files = { 'train' : lst_ns[start:],
                                'test'  : lst_ns[:start]}

                processes = [
                        FeatMultiProcsClass(
                                i, f"Thread-{i}",
                                sub_src[i],
                                train_set,
                                ntype,
                                noise_files,
                                snr_db,
                                garb_files,
                                success_dict,
                                params_audio_def = params_audio)
                                    for i in range(args.num_procs)]

                start_time = time.time()

                if DEBUG:
                    for proc in processes:
                        proc.run()
                else:
                    for proc in processes:
                        proc.start()

                    for proc in processes:
                        proc.join()
                    print(f"Time elapse {time.time() - start_time} sec")

                if args.wandb_track:
                    data = wandb.Artifact(
                        S3_BUCKET + "-tfrecords",
                        type="dataset",
                        description="tfrecords of speech command dataset")
                    data.add_reference(f"s3://{S3_BUCKET}/{S3_PREFIX}", max_objects=31000)
                    run.log_artifact(data)

                for lst in success_dict.values():
                    tot_success_dict[train_set] += lst

    if not DEBUG:
        for train_set in train_sets:
            with open(f'data/{train_set}_tfrecords_s2i.csv', 'w') as file: # pylint: disable=unspecified-encoding
                for tfrecord in tot_success_dict[train_set]:
                    tfrecord = re.sub(r'\\', '/', tfrecord)
                    file.write(f'{tfrecord}\n')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Generate TFrecord formatted input data from a raw speech commands dataset')

    argparser.add_argument(
        '-t',
        '--train_dataset_path',
        default = 'data/train_files_s2i.csv',
        help    = 'path to train data file')

    argparser.add_argument(
        '--test_dataset_path',
        default = 'data/test_files_s2i.csv',
        help    = 'path to test data file')

    argparser.add_argument(
        '-n',
        '--num_procs',
        type    = int,
        default = 2,
        help='How many processor cores to use for execution')

    argparser.add_argument(
        '-w',
        '--wandb_track',
        default = False,
        help    = 'Enable tracking of this run in Weights&Biases')

    argparser.add_argument(
        '--wandb_project',
        type    = str,
        default = 'speech-to-intent',
        help='Weights&Biases project name')

    argparser.add_argument(
        '--wandb_entity',
        type    = str,
        default = 'ambiq',
        help    = 'Weights&Biases entity name')

    args_ = argparser.parse_args()
    main(args_)
