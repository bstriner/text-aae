import tensorflow as tf
from tensorflow.contrib import slim

from text_aae.sn.sn_linear import SNLinear
from ...sn.sn_conv1d import SNConv1D


def make_discriminator_bigan_cnn_fn(bn=False):
    def discriminator_bigan_cnn_fn(x, z, params, is_training=True):
        dis = DiscriminatorBiganCnn(params=params, bn=bn)
        return dis.call(x, z, is_training=is_training)

    return discriminator_bigan_cnn_fn


class DiscriminatorBiganCnn(object):
    def __init__(self, params, bn=False):
        self.embedding = SNLinear(
            units=params.discriminator_dim,  # params.feature_dim
            name='discriminator_projection'
        )
        self.layers = [
            SNConv1D(
                filters=params.discriminator_dim,
                kernel_size=7,
                padding='same',
                data_format='channels_last',
                name='discriminator_conv1d_{}'.format(i),
                activation=tf.nn.leaky_relu
            )
            for i in range(6)
        ]
        self.projection = SNLinear(
            units=1,  # params.feature_dim
            name='discriminator_projection'
        )
        self.bn = bn

    def call(self, x, z, is_training=True):
        xembed = self.embedding(x)
        h = tf.concat((xembed, z), axis=-1)
        for l in self.layers:
            h = l(h)
            if self.bn:
                h = slim.batch_norm(h, is_training=is_training)
        h = tf.reduce_mean(h, axis=1)
        y = self.projection(h)
        return y, h
