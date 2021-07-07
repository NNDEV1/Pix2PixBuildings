import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
from PIL import Image, ImageOps
from utils import get_shape, batch_norm, lkrelu

class Discriminator(object):
    def __init__(self, inputs, is_training, stddev=0.02, center=True, scale=True, reuse=None):
        self._is_training = is_training
        self._stddev = stddev

        with tf.variable_scope('D', initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._center = center
            self._scale = scale
            self._prob = 0.5
            self._inputs = inputs
            self._discriminator = self._build_discriminator(inputs, reuse=reuse)

    def build_layer(self, name, inputs, k, bn=True, use_dropout=False):
        layer = dict()

        with tf.variable_scope(name):
            layer['filters'] = tf.get_variable('filters', [4, 4, get_shape(inputs)[-1], k])
            layer['conv'] = tf.nn.conv2d(inputs, layer['filters'], strides=[1, 2, 2, 1], padding='SAME')
            layer['bn'] = batch_norm(layer['conv'], center=self._center, scale=self._scale, training=self._is_training) if bn else layer['conv']
            layer['dropout'] = tf.nn.dropout(layer['bn'], self._prob) if use_dropout else layer['bn']
            layer['fmap'] = lkrelu(layer['dropout'], slope=0.2)

        return layer

    def _build_discriminator(self, inputs, reuse=None):
        discriminator = dict()

        discriminator['l1'] = self.build_layer('l1', inputs, 64, bn=False)
        discriminator['l2'] = self.build_layer('l2', discriminator['l1']['fmap'], 128)
        discriminator['l3'] = self.build_layer('l3', discriminator['l2']['fmap'], 256)
        discriminator['l4'] = self.build_layer('l4', discriminator['l3']['fmap'], 512)

        with tf.variable_scope('15'):
            l5 = dict()
            l5['filters'] = tf.get_variable('filters', [4, 4, get_shape(discriminator['l4']['fmap'])[-1], 1])
            l5['conv'] = tf.nn.conv2d(discriminator['l4']['fmap'], l5['filters'], strides=[1, 1, 1, 1], padding="SAME")
            l5['bn'] = batch_norm(l5['conv'], center=self._center, scale=self._scale, training=self._is_training)
            l5['fmap'] = tf.nn.sigmoid(l5['bn'])
            discriminator['l5'] = l5

        return discriminator
