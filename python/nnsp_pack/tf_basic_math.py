"""
Basic tensorflow math functions
"""
import tensorflow as tf
def tf_log10_eps(val, eps = 2.0**-15):
    """
    log10 with minimum eps
    """
    return  tf.math.log(tf.maximum(eps, val)) / tf.math.log(10.0)
