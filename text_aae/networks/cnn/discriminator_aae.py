import tensorflow as tf
from tensorflow.contrib import slim

from text_aae.sn.sn_linear import SNLinear
from ...sn.sn_conv1d import SNConv1D


def make_discriminator_aae_cnn_fn(bn=True):
    def discriminator_aae_cnn_fn(z, params, is_training=True):
        dis = DiscriminatorAaeCnn(params=params, bn=bn)
        return dis.call(z, is_training=is_training)

    return discriminator_aae_cnn_fn


class DiscriminatorAaeCnn(object):
    def __init__(self, params, bn=True):
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

    def call(self, z, is_training=True):
        h = z
        for l in self.layers:
            h = l(h)
            if self.bn:
                h = slim.batch_norm(h, is_training=is_training)
        h = tf.reduce_mean(h, axis=1)
        y = self.projection(h)
        return y, h
