import tensorflow as tf

from .resnet import building_block_v2
from ...sn.sn_conv1d import sn_conv1d
from ...sn.sn_linear import sn_linear


def build_projection(inputs, params, activation_fn=tf.nn.relu, bn_fn=None, is_training=True):
    h = inputs
    h = sn_linear(
        inputs=h,
        units=params.discriminator_dim,
        activation=None,
        name='projection_linear_0')
    if bn_fn:
        h = bn_fn(h, is_training=is_training, scope='projection_bn_0')
    h = activation_fn(h)
    h = sn_linear(
        inputs=h,
        units=1,
        activation=None,
        name='projection_linear_1')
    h = tf.reduce_mean(h, axis=1)
    return h


def norm(x):
    return x / tf.maximum(tf.norm(x, ord=2, axis=-1, keep_dims=True), 1)


def make_discriminator_gan_cnn_ml_resnet_fn(
        bn_fn=None,
        padding='same',
        kernel_size=7,
        layers=6,
        emedding_scale=1.,
        activation_fn=tf.nn.leaky_relu):
    def dis_fn(x, params, is_training=True):
        h = x
        h = sn_linear(
            inputs=h,
            units=params.discriminator_dim,
            activation=None,
            name='discriminator_input'
        ) * emedding_scale
        # h = norm(h)
        projections = []
        for i in range(layers):
            with tf.variable_scope('discriminator_projection_{}'.format(i)):
                projections.append(build_projection(
                    inputs=h,
                    params=params,
                    is_training=is_training,
                    activation_fn=activation_fn,
                    bn_fn=bn_fn))
            with tf.variable_scope('discriminator_block_{}'.format(i)):
                h = building_block_v2(
                    inputs=h,
                    filters=params.discriminator_dim,
                    kernel_size=kernel_size,
                    activation_fn=activation_fn,
                    conv_fn=sn_conv1d,
                    data_format='NHWC',
                    bn_fn=bn_fn,
                    padding=padding,
                    training=is_training,
                    strides=1,
                    delta_weight=0.2,
                    shortcut_weight=1,
                    projection_shortcut=None)
                # h = norm(h)
        with tf.variable_scope('discriminator_projection_final'):
            projections.append(build_projection(
                inputs=h,
                params=params,
                is_training=is_training,
                activation_fn=activation_fn,
                bn_fn=bn_fn))

        total = tf.add_n(projections)
        return total, h

    return dis_fn
