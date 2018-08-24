import tensorflow as tf


def softplus_rect(x, eps=1e-6):
    return eps + tf.nn.softplus(x)


def sample(mu, sigma):
    rnd = tf.random_normal(shape=tf.shape(mu))
    return mu + (rnd * sigma)


def kl_divergence(mu, sigma):
    kl = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, -1)
    return kl
