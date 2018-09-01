import tensorflow as tf

from text_aae.sn.sn_linear import SNLinear
from ...sn.sn_conv1d import SNConv1D


def make_discriminator_gan_cnn_ml_fn(
        bn_fn=None,
        activation_fn=tf.nn.leaky_relu,
        padding='same',
        kernel_size=7,
        layers=6,
        emedding_scale=1.):
    def discriminator_gan_cnn_fn(x, params, is_training=True):
        dis = DiscriminatorGanCnnMl(
            params=params,
            bn=bn_fn,
            padding=padding,
        activation_fn=activation_fn,
            kernel_size=kernel_size,
            layers=layers,
            emedding_scale=emedding_scale)
        return dis.call(x, is_training=is_training)

    return discriminator_gan_cnn_fn


class Projection(object):
    def __init__(self, params, name, bn=None, activation_fn=tf.nn.leaky_relu):
        self.layers = [
            SNLinear(
                units=params.discriminator_dim,
                name='{}_projection_0'.format(name),
                activation=activation_fn
            ),
            SNLinear(
                units=1,  # params.feature_dim
                name='{}_projection_1'.format(name),
                activation=None
            )]
        self.name = name
        self.bn = bn

    def call(self, x, is_training=True):
        with tf.variable_scope(self.name):
            h = x
            h = self.layers[0](h)
            if self.bn is not None:
                h = self.bn(h, is_training=is_training, scope='{}_batch_norm'.format(self.name))
            h = self.layers[1](h)
            h = tf.reduce_mean(h, axis=1)
            return h


class DiscriminatorGanCnnMl(object):
    def __init__(self, params, bn=None, kernel_size=7, layers=6, padding='same',
                 activation_fn=tf.nn.leaky_relu,
                 emedding_scale=1.):
        self.embedding = SNLinear(
            units=params.discriminator_dim,  # params.feature_dim
            name='discriminator_embedding'
        )
        self.layers = [
            SNConv1D(
                filters=params.discriminator_dim,
                kernel_size=kernel_size,
                padding=padding,
                data_format='channels_last',
                name='discriminator_conv1d_{}'.format(i),
                activation=activation_fn
            )
            for i in range(layers)
        ]
        self.projections = [
            Projection(
                params=params,
                name='projection_{}'.format(i),
                activation_fn=activation_fn,
                bn=bn)
            for i in range(layers + 1)]
        self.bn = bn
        self.activation_fn=activation_fn

        self.emedding_scale = emedding_scale

    def call(self, x, is_training=True):
        h = x
        h = self.embedding(h) * self.emedding_scale
        projections = []
        for i, l in enumerate(self.layers):
            projection = self.projections[i].call(h, is_training=is_training)
            projections.append(projection)
            h = l(h)
            if self.bn is not None:
                h = self.bn(h, is_training=is_training, scope='discriminator_bn_{}'.format(i))
        projection = self.projections[-1].call(h, is_training=is_training)
        projections.append(projection)

        total = tf.add_n(projections)

        return total, h
