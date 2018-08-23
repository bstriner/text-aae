import tensorflow as tf


def fm_losses(h_real, h_fake, **kwargs):
    assert len(h_real.shape) == 2
    ey_real = tf.reduce_mean(h_real, 0)
    ey_fake = tf.reduce_mean(h_fake, 0)
    dist = tf.reduce_sum(tf.square(ey_fake - ey_real))
    gloss = dist
    dloss = -dist
    return gloss, dloss


def wgan_losses(y_real, y_fake, **kwargs):
    assert len(y_real.shape) == 2
    ey_real = tf.reduce_mean(y_real)
    ey_fake = tf.reduce_mean(y_fake)
    dloss = ey_fake - ey_real
    gloss = ey_real - ey_fake
    return gloss, dloss


def gan_losses(y_real, y_fake, **kwargs):
    assert len(y_real.shape) == 2
    dis_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(y_real),
        logits=y_real,
        name='dis_loss_real'
    )
    dis_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(y_fake),
        logits=y_fake,
        name='dis_loss_fake'
    )
    gen_loss = tf.nn.softplus(-y_fake)

    gloss = tf.reduce_mean(gen_loss)
    dloss = tf.reduce_mean(dis_loss_fake) + tf.reduce_mean(dis_loss_real)

    return gloss, dloss


def sign_sq(x):
    return tf.square(x) * tf.sign(x)


def gan_losses_sq(y_real, y_fake, **kwargs):
    assert len(y_real.shape) == 2
    ey_real = tf.reduce_mean(y_real)
    ey_fake = tf.reduce_mean(y_fake)
    dloss = sign_sq(ey_fake - ey_real)
    gloss = sign_sq(ey_real - ey_fake)
    return gloss, dloss


def gan_losses_h(y_real, y_fake, h_real, h_fake):
    assert len(y_real.shape) == 2
    assert len(h_real.shape) == 2
    ey_real = tf.reduce_mean(y_real)
    ey_fake = tf.reduce_mean(y_fake)
    eh_real = tf.reduce_mean(h_real, axis=0)
    eh_fake = tf.reduce_mean(h_fake, axis=0)
    dloss = ey_fake - ey_real
    gloss = tf.reduce_sum(tf.square(eh_real - eh_fake))
    return gloss, dloss
