import tensorflow as tf
from tensorflow.contrib import slim

from ..embedding import embedding_fn


def encoder_cnn_fn(x, vocab_size, params, is_training=True):
    h = x
    h = embedding_fn(
        h,
        vocab_size=vocab_size,
        dim_out=params.encoder_dim,
        name='encoder_embedding')
    for i in range(6):
        h = slim.conv1d(
            h,
            num_outputs=params.encoder_dim,
            kernel_size=7,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_conv1d_{}'.format(i))
        #h = slim.batch_norm(h, is_training=is_training)
    mu = slim.fully_connected(h, num_outputs=params.latent_dim, activation_fn=None, scope='encoder_mu')
    logsigma = slim.fully_connected(h, num_outputs=params.latent_dim, activation_fn=None, scope='encoder_logsigma')
    #sigma = tf.nn.softplus(logsigma)
    #rnd = tf.random_normal(shape=tf.shape(logsigma))
    #z = tf.add(mu, rnd * sigma, name='encoder_z')
    #return z
    return mu, logsigma
