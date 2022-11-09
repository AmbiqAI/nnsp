# NN Speech
NN Speech (NNSP) integrates 3 neural networks, speech activitiy detection, HI-Galaxy and Speech to intent (S2I). We start from S2I model as an example.
## Directory contents
```py
nnsp/ # root 
    evb/ # for evb deployment
        build/      # bin files
        includes/   # required inlcudes
        libs/       # required libs
        make/       # make.mk
        pack/
        src/        # c source codes
        Makfile
        autogen.mk
    ns-nnsp/  # c codes to build nnsp library (used only when re-building library)
    python/   # for NN training
    README.md # this readme
```
## Prerequisite
### `Software`
To work on Apollo4, you need
- Arm GNU Toolchain 11.3
- Segger J-Link v7.56+
### `Dataset`
You need to download several datasets. Please read the license carefully. The detail can be found [here](./docs/README.md).
# Run a RNN-based Speech to Intent Model
The original example can be found in AIWinterMute's S2I model, forked [here](https://github.com/AIWintermuteAI/Speech-to-Intent-Micro), though significantly re-written to work with AmbiqSuite on Ambiq's EVBs. The NN model is based on feed-forward NN, where each layer utilizes fully-connected (FC) layer, convolutional neural network (CNN) layer or residue net (resnet).

There are few disadvantages in the original model:
1. No out-of-vocabulary (OOV) data in the training data. In words, when you speak something irrelevant to the desired comments, it still gives you a results, which is kind of false alarm (FA)
2. It is not a real-time processing. You need to press a button to let the machine to know that data (speech) is comming in. Then it will accumulate few seconds to run the inference.
3. Since it requires to accumulate the data for few seconds, we need to decide what duration of the input data we should use when we train the model. As we know that the speed of people's utterances for the same sentence is very dynamic, sometimes it is hard to decide this hyper-parameter. One simplest way is to choose the longest pronuciation of your training data. But it might be not that practical in certain cases. 

Alternatively, we provide a solution to overcome those issues in this example. We apply recurrent neural net (RNN) to our model. RNN is a very natural way to deal with real-time process data and dynamic length of different pronunciations. 

## Compiling and Running a Pre-Trained Model

From the `nnsp/evb` directory:

1. `make clean`
2. `make` or `make BOARD=apollo4b` depending on board type
3. `make deploy` Ensure your board is connected via the JLINK USB port and
   turned on first.
4. Plug a mic into the 3.5mm port, and push BTN0 to initiate voice recording
5. `make view` will provide SWO output as the device is running, showing 
   predicted slots/intents etc.

`Note`: Due to the authority of the third-part licenses related to the training datasets (see [here](docs/README.md)), we couldn't provide a well-trained model here. The weight tables of NN we deployed on evb here is just in a random number. And the result is basically incorrect. Once you have the permission to access the data. please download it and train the model as  the procedure below. 
## Re-Training a New Model

Our approach to training the model can be found in [README.md](./python/README.md). The trained model is saved in [evb/src/def_nn0_s2i.c](evb/src/def_nn0_s2i.c) and [evb/src/def_nn0_s2i.h](evb/src/def_nn0_s2i.h). 

## Library NS-NNSP Library Overview
Library neuralspot NNSP, `ns-nnsp.a`, is a C library to build a pipeline including feature extraction and neural network to run on Apollo4. The source code is under the folder `ns-nnsp/`. You can modify or rebuild it via [NeuralSPOT Ambiq's AI Enablement Library](https://github.com/AmbiqAI/neuralSPOT).
In brief, there are two basic building blocks inside `ns-nnsp.a`, feature extraction and neural network. In `ns-nnsp.a`, we call them `FeatureClass` defined in `feature_module.h` and `NeuralNetClass` in `neural_nets.h`, respectively. Furthermore, `NNSPClass` in `nn_speech.h` encapsulates them to form a concrete instance.
We illustrate this in Fig. 1. 
<p align="center">
  <img src="./pics/nnsp_flow.jpg"  width="80%">
</p>
<p align="center">
  Fig. 1: Illustration of `ns-nnsp`
</p>

Also, in our specific s2i NN case, `def_nn0_s2i.c` has two purposes:
  1. For feature extraction, we use Mel spectrogram with 40 Mel-scale. To apply the standarization to the features in training dataset, it requires statistical mean and standard deviation, which is defined in `def_nn0_s2i.c`. 
  2. For the neural network, it points to the trained weight table defined in `def_nn0_s2i.c` as well.

# NNSP Model: speech activity detection + Hi-Galaxy + S2I
If there are more NNs, users need to define the other instances of `NNSPClass` and write a control among them, depending on users' applications. 

One common example for the speech application might be running a `speech activity detection` and `Hi-Galaxy` in front of the S2I model. `speech activity detection` is used to detect whether speech is present or not. Usually, the size of `speech activity detection` model is smaller. Hence, it provides one approch to saving memory and power. After `speech activity detection` assures the speech is observed, some more complicated processes will proceed further. 

`Wakeup-Keyword` is widespread in our life nowadays, e.g., `Hey-Siri` for iPhones, `Alexa` for Amazon Echo devices, etc. If the device detects its name, it will do further processes, such as voice commands or even connecting to the cloud to search whatever users want to do.
In our case, we use `Hi-Galaxy` as our keyword, where you can download the dataset from [here](https://developer.qualcomm.com/project/keyword-speech-dataset) `[1]`.

In our example, we control 3 NNs, `speech activity detection`, `Hi-Galaxy` and `S2I` sequentially.

`[1]` Byeonggeun Kim, Mingu Lee, Jinkyu Lee, Yeonseok Kim, and Kyuwoong Hwang, “Query-by-example on-device keyword spotting,” to be published in IEEE Automatic Speech Recognition and Understanding Workshop (ASRU 2019), Sentosa, Singapore, Dec. 2019 to be published


## Compiling and Running a Pre-Trained '`speech activity detection`'+KWS+S2I Model

1. Go to `nnsp/evb/` directory
1. `make clean`
1. `make NNSP_MODE=1` or `make BOARD=apollo4b NNSP_MODE=1` depending on board type
1. `make deploy NNSP_MODE=1` Ensure your board is connected via the JLINK USB port and
   turned on first.
1. Plug a mic into the 3.5mm port, and push BTN0 to initiate voice recording
1. `make view` will provide SWO output as the device is running, showing 
   predicted slots/intents etc.

`Note`: Due to the authority of the third-part licenses related to the training datasets (see [here](docs/README.md)), we couldn't provide a well-trained model here. The weight tables of NN we deployed on evb here is just in a random number. And the result is basically incorrect. Once you have the permission to access the data. please download it and train the model (see [here](./python/README.md)). 
# Build NS-NNSP library from NeuralSPOT (Optional)
If you want to modify or re-build the `ns-nnsp.a` library, you can follow the steps here. 
1. Download NeuralSPOT
```bash
$ git clone https://github.com/AmbiqAI/neuralSPOT.git ../neuralSPOT
```
2. Copy the source code of NS-NNSP to NeuralSPOT. Then go to NeuralSPOT folder.
```bash
$ cp -a ns-nnsp ../neuralSPOT/neuralspot; cd ../neuralSPOT
```
3. Open `neuralSPOT/Makefile` and append the `ns-nnsp` to the library modules as below
```bash
# NeuralSPOT Library Modules
modules      := neuralspot/ns-harness 
modules      += neuralspot/ns-peripherals 
modules      += neuralspot/ns-ipc
modules      += neuralspot/ns-audio
modules      += neuralspot/ns-usb
modules      += neuralspot/ns-utils
modules      += neuralspot/ns-rpc
modules      += neuralspot/ns-i2c
modules      += neuralspot/ns-nnsp # <---add this line

# External Component Modules
modules      += extern/AmbiqSuite/$(AS_VERSION)
modules      += extern/tensorflow/$(TF_VERSION)
modules      += extern/SEGGER_RTT/$(SR_VERSION)
modules      += extern/erpc/$(ERPC_VERSION)
```
4. Compile
```bash
$ make clean; make; make nest
```
5. Copy the necessary folders back to `nnsp` folder
```bash
$ cd nest; cp -a pack includes libs ../nnsp/evb
```