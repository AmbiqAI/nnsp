"""
This module deals with sythesizing speech and noise data
"""
import os
import re
import logging
import soundfile as sf
import numpy as np
import librosa

def get_power(data):
    """Calculate power of data"""
    return np.mean(data**2)

def add_noise(data, noise, snr_db, stime, etime):
    """Synthesize noise and speech"""
    pw_data = get_power(data[stime:etime])
    pw_noise = get_power(noise)
    snr = 10**(snr_db/10)
    if pw_data != 0:
        data = data / np.sqrt(pw_data)
    if pw_noise != 0 and snr != 0:
        noise = noise / np.sqrt(pw_noise) / np.sqrt(snr)
    output = data + noise
    max_val = np.abs(output).max()
    prob = np.random.uniform(0.05, 0.95, 1)

    output = (output / (max_val + 10**-5)) * prob
    return output

def get_noise_files_new(path_noise_folder):
    """Fetch all of noise files"""
    lst = []
    for root, _, files in os.walk(f'wavs/noise/{path_noise_folder}'):
        for file in files:
            if re.search(r'wav$', file):
                lst += [os.path.join(root, file.strip())]
    return lst

def get_noise_files(files_list, noise_type):
    """Fetch all of noise files"""
    lst = []
    for root, _, files in os.walk(f'wavs/noise/{noise_type}/{files_list}'):
        for file in files:
            if re.search(r'wav$', file):
                lst += [os.path.join(root, file.strip())]
    return lst

def get_noise(fnames, length = 16000 * 5):
    """Random pick ONE of noise from fnames"""
    len0 = len(fnames)
    rand_idx = np.random.randint(0, len0)

    try:
        noise, sample_rate_in = sf.read(fnames[rand_idx])
    except: # pylint: disable=W0702
        logging.debug('reading noise file %s failed', fnames[rand_idx] )
        noise = np.random.randn(16000).astype(np.float32) * 0.1
    else:
        if noise.ndim > 1:
            noise = noise[:,0]

        if sample_rate_in > 16000:
            try:
                noise = librosa.resample(
                            noise,
                            orig_sr=sample_rate_in,
                            target_sr=16000)
            except: # pylint: disable=W0702
                logging.debug('resampling noise %s failed. Loading random noise',  fnames[id])
                noise = np.random.randn(length).astype(np.float32) * 0.1
        elif sample_rate_in < 16000:
            logging.debug('reading noise file %s sampling rate < 16000', fnames[rand_idx])
            noise = np.random.randn(length).astype(np.float32) * 0.1

        if len(noise) > length:
            start = np.random.randint(0, len(noise)-length)
            noise = noise[start : start+length]
        else:
            repeats = int(np.ceil(length / len(noise)))
            noise = np.tile(noise, repeats)
            noise = noise[:length]

    return noise
