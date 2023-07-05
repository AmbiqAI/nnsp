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
SAMPLES = 1000
DIM_INPUT = 100
DIM_OUTPUT=50
# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(
            low     = 0,
            high    = 2 * math.pi,
            size    = (SAMPLES, DIM_INPUT)).astype(np.float32)

y_values = np.random.uniform(
            low     = 0,
            high    = 2 * math.pi,
            size    = (SAMPLES, DIM_OUTPUT)).astype(np.float32)

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
model.add(keras.layers.Dense(100, activation='tanh', input_shape=(DIM_INPUT,)))
model.add(keras.layers.Dense(DIM_OUTPUT, activation="relu"))
model.add(keras.layers.Dense(DIM_OUTPUT, activation="sigmoid"))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#%% Train the mdoel
# Train the model on our training data while validating on our validation set
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                        validation_data=(x_validate, y_validate))
# Save the model to disk
model.save(MODEL_TF)
#%%
def representative_dataset():
    """
    estimate data input
    """
    for i in range(500):
        val = np.random.uniform(
            low     = -2**15 / 2.0**8,
            high    = (2**15-1) / 2.0**8,
            size    = (1, DIM_INPUT)).astype(np.float32)
        yield([val.reshape(1, -1)]) # pylint: disable=superfluous-parens

# Set the optimization flag.
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
converter.inference_input_type  = tf.int16
converter.inference_output_type = tf.int16
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open(MODEL_TFLITE, "wb").write(model_tflite)
# Convert the weight table to c code
os.system(f"xxd -i {MODEL_TFLITE} > quant_model_act.h")
