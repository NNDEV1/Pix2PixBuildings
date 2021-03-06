import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image, ImageOps

def get_shape(tensor):
    return tensor.get_shape().as_list()

def batch_norm(*args, **kwargs):

    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)

    return bn

def lkrelu(x, slope=0.01):
    return tf.maximum(slope*x, x)
