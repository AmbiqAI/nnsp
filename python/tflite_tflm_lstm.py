"""
tflite to tflm converter
"""
import os
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np

MODELS_DIR = 'model_tflite/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'model'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

# Number of sample datapoints
EPOCHS      = 1
BITS_ACT    = 16
SAMPLES     = 1000
DIM_INPUT   = 72
DIM_OUTPUT  = 257
STEPS       = 6
BATCH_SIZE  = 1
# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(
            low     = 0,
            high    = 2 * math.pi,
            size    = (SAMPLES,STEPS,DIM_INPUT,1)).astype(np.float32)

y_values = np.random.uniform(
            low     = 0,
            high    = 2 * math.pi,
            size    = (SAMPLES,STEPS, DIM_OUTPUT,1)).astype(np.float32)

# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT  = int(0.2 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT]) # pylint: disable=unbalanced-tuple-unpacking
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT]) # pylint: disable=unbalanced-tuple-unpacking

#%% Build the model
# We'll use Keras to create a simple model architecture
model = tf.keras.Sequential()
model.add(keras.layers.Input(
            batch_size  = BATCH_SIZE,
            shape       = (STEPS, DIM_INPUT, 1), # (steps, dim_feat, chs)
            name        = "input_compmel",
            dtype       = np.float32)) # Input tensor
model.add(keras.layers.Conv2D(
            72,
            (6, 72),
            padding     = 'valid',
            strides     = (1, 1), # downsampling 2 in timesteps dim
            activation  = 'tanh',
            input_shape = (None, 72, 1)) # (time, dim_feat, ch)
)
model.add(keras.layers.Reshape((1, -1)))
model.add(keras.layers.Dense(72, activation='tanh'))
model.add(keras.layers.LSTM(
            72,
            return_sequences = True,
            return_state = False,
            stateful = False,
            unit_forget_bias = True,
            activation='tanh',
            recurrent_activation='sigmoid',
            unroll=True))
model.add(keras.layers.Dense(72, activation="relu"))
model.add(keras.layers.Dense(72, activation="relu"))
model.add(keras.layers.Dense(72, activation="relu"))
model.add(keras.layers.Dense(DIM_OUTPUT, activation="sigmoid"))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#%% Train the mdoel
# Train the model on our training data while validating on our validation set
history = model.fit(
        x_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (x_validate, y_validate))
# Save the model to disk
model.save(MODEL_TF)
#%%

def representative_dataset():
    """
    estimate data input
    """
    for _ in range(500):
        val = np.random.uniform(
            low     = -2**BITS_ACT / 2.0**(BITS_ACT>>1),
            high    = (2**BITS_ACT-1) / 2.0**(BITS_ACT>>1),
            size    = (1, 6, DIM_INPUT, 1)).astype(np.float32)
        yield {"input_compmel" : val}

# Set the optimization flag.
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
if BITS_ACT==8:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
elif BITS_ACT==16:
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ] # enable TensorFlow ops.
    converter.inference_input_type  = tf.int16
    converter.inference_output_type = tf.int16
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open(MODEL_TFLITE, "wb").write(model_tflite)
# Convert the weight table to c code
os.system(f"xxd -i {MODEL_TFLITE} > quant_model_act.h")
