import tensorflow as tf


def fm_losses(y_real, y_fake):
    assert len(y_real.shape) == 2
    ey_real = tf.reduce_mean(y_real, 0)
    ey_fake = tf.reduce_mean(y_fake, 0)
    dist = tf.reduce_sum(tf.square(ey_fake - ey_real))
    gloss = dist
    dloss = -dist
    return gloss, dloss
