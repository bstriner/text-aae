import tensorflow as tf

from text_aae.sn.sn_linear import SNLinear
from ...sn.sn_conv1d import SNConv1D


def discriminator_cnn_fn(z, params, is_training=True):
    dis = CnnDiscriminator(params=params)
    return dis.call(z, is_training=is_training)


class CnnDiscriminator(object):
    def __init__(self, params):
        self.layers = [
            SNConv1D(
                filters=params.discriminator_dim,
                kernel_size=7,
                padding='same',
                data_format='channels_last',
                name='discriminator_conv1d_{}'.format(i)
            )
            for i in range(6)
        ]
        self.projection = SNLinear(
            num_units=1,  # params.feature_dim
            name='discriminator_projection'
        )

    def call(self, z, is_training=True):
        h = z
        for l in self.layers:
            h = l(h)
        h = tf.reduce_mean(h, axis=1)
        y = self.projection(h)
        return y, h
