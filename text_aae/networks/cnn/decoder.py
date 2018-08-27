import tensorflow as tf
from tensorflow.contrib import slim


def make_decoder_cnn_fn(bn=True):
    def decoder_cnn_fn(z, vocab_size, params, is_training=True):
        h = z
        for i in range(6):
            h = slim.conv1d(
                h,
                num_outputs=params.decoder_dim,
                kernel_size=7,
                activation_fn=tf.nn.leaky_relu,
                scope='decoder_conv1d_{}'.format(i))
            if bn:
                h = slim.batch_norm(h, is_training=is_training, scope='decoder_bn_{}'.format(i))
        logits = slim.fully_connected(
            h,
            num_outputs=vocab_size,
            activation_fn=None,
            scope='decoder_logits'
        )
        return logits

    return decoder_cnn_fn
