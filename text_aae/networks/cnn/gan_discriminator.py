import tensorflow as tf

from text_aae.sn.sn_linear import SNLinear
from ...sn.sn_conv1d import SNConv1D


def gan_discriminator_cnn_fn(x, params, is_training=True):
    dis = GanCnnDiscriminator(params=params)
    return dis.call(x, is_training=is_training)


class GanCnnDiscriminator(object):
    def __init__(self, params):
        self.embedding = SNLinear(
            num_units=params.discriminator_dim,  # params.feature_dim
            name='discriminator_embedding'
        )
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

    def call(self, x, is_training=True):
        h = x
        h = self.embedding(h)
        for l in self.layers:
            h = l(h)
        h = tf.reduce_mean(h, axis=1)
        y = self.projection(h)
        return y, h
