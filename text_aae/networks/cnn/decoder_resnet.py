import tensorflow as tf
from tensorflow.contrib import slim

from .resnet import building_block_v2


def make_decoder_resnet_cnn_fn(bn=True, kernel_size=7, layers=6, padding='SAME', activation_fn=tf.nn.relu):
    def decoder_cnn_fn(z, vocab_size, params, is_training=True):
        h = z
        h = slim.fully_connected(
            h, num_outputs=params.decoder_dim,
            activation_fn=None,
            scope='decoder_input'
        )
        for i in range(layers):
            with tf.variable_scope('decoder_resnet_{}'.format(i)):
                h = building_block_v2(
                    inputs=h,
                    filters=params.decoder_dim,
                    kernel_size=kernel_size,
                    activation_fn=activation_fn,
                    training=is_training,
                    conv_fn=slim.conv1d,
                    data_format='NHWC',
                    projection_shortcut=None,
                    strides=1,
                    bn_fn= slim.batch_norm if bn else None,
                    padding=padding
                )
        logits = slim.fully_connected(
            h,
            num_outputs=vocab_size,
            activation_fn=None,
            scope='decoder_logits'
        )
        return logits

    return decoder_cnn_fn
