"""
Module to calculate loss function
"""
import tensorflow as tf
def cross_entropy(target,       # trget
                  estimation,   # estimator
                  masking,      # mask
                  eps = tf.constant(2**-15, dtype = tf.float32)): # epsilon
    """Cross entropy loss"""
    loss = -tf.reduce_sum(
        masking * target * tf.math.log(tf.math.maximum(estimation, eps)) )
    steps = tf.reduce_sum(masking)
    ave_loss = loss / steps
    return ave_loss, steps

def loss_mse(
        target,       # trget
        estimation,   # estimator
        masking     # mask
        ): # epsilon
    """MSE loss"""
    loss = tf.reduce_sum(
        masking * tf.math.square(target - estimation) )
    steps = tf.reduce_sum(masking)
    ave_loss = loss / steps
    return ave_loss, steps