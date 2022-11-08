1. `Fluent speech commands` \
To download the Fluent Speech command dataset, please read the full license carefully in [Fluent Speech Commands Public License](https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf). Download and extract the data and put it under the folder `nnsp/python/wavs/` as shown in Table 1.
1. `LibriSpeech 100-hour and 360-hour ASR corpus` \
LibriSpeech ASR corpus is used to make out-of-vocabulary (OOV) data and can be downloaded from [here](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and [here](https://www.openslr.org/resources/12/train-clean-360.tar.gz) for 100-hour and 360-hour clean datasets, respectively.
Download and extract the data and put it under the folder `nnsp/python/wavs/garb/en/` as shown in Table 1.
1. `Qualcomm Keyword Speech Dataset` \
Please read the license carefully, which can be found [here](./LICENSE.pdf) or [here](
https://developer.qualcomm.com/project/keyword-speech-dataset). There are 4 keywords available there. We only use Hi-Galaxy. Download and extract the data to the folder named `qualcomm_keyword_speech_dataset` and put it under the folder `nnsp/python/wavs/kws/` as shown in Table 1.
1. `MUSAN dataset` \
The MUSAN (A Music, Speech, and Noise Corpus) dataset can be download from [here](
http://www.openslr.org/17/). 
Download and extract the data to the folder named `musan` and put it under the folder `nnsp/python/wavs/noise/` as shown in Table 1.
1. `THCHS-30 dataset` \
THCHS30 is an open Chinese speech database published by Center for Speech and Language Technology (CSLT) at Tsinghua University. You can download the data from [here](
https://www.openslr.org/resources/18/data_thchs30.tgz).
Download and extract the data to the folder named `data_thchs30` and put it under the folder `nnsp/python/wavs/garb/cn/` as shown in Table 1.
```py
nnsp/ # root 
    evb/ 
    ns-nnsp/  
    python/   
        wavs/
            speakers/ # (1) Fluent speech commands dataset 
            garb/
                en/LibriSpeech/
                    train-clean-100/ # (2)--LibriSpeech 100-hour ASR corpus
                    train-clean-360/ # (2)--LibriSpeech 360-hour ASR corpus
                cn/data_thchs30/    # (5)--THCHS-30 dataset
            kws/qualcomm_keyword_speech_dataset # (3)--Qualcomm Keyword Speech Dataset
            noise/musan/ # (4)--MUSAN dataset
    README.md 
```
<p align="center">
  Table 1: Illustration of NNSP
</p>
