import numpy as np
import tensorflow as tf


def power_iter(u, w):
    vt = tf.nn.l2_normalize(tf.matmul(tf.transpose(w, (1, 0)), u))
    ut = tf.nn.l2_normalize(tf.matmul(w, vt))
    return ut, vt


# def spec_norm(u, w, v):
#    s = tf.matmul(tf.matmul(tf.transpose(u, (1, 0)), w), v)
#    ws = w / s
#    return ws

def spec_norm(u, w):
    return tf.norm(tf.matmul(tf.transpose(w, (1, 0)), u), ord=2)


def sn_calc(w):
    shape = [s.value for s in w.shape]
    dimin = np.prod(shape[:-1])
    dimout = shape[-1]
    w = tf.reshape(w, (dimin, dimout))
    u = tf.get_variable(
        name='u',
        shape=(dimin, 1),
        initializer=tf.initializers.random_normal(0.5),
        dtype=tf.float32,
        trainable=False)

    ut, vt = power_iter(u, w)
    if not tf.get_variable_scope().reuse:
        uup = tf.assign(u, ut, name='update_u')
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, uup)

    sn = spec_norm(ut, w)
    sn = tf.maximum(sn, 1.)
    return sn


def sn_kernel(shape, scope, init=tf.initializers.glorot_normal()):
    with tf.variable_scope(scope):
        w = tf.get_variable(name='w_raw', shape=shape, initializer=init, dtype=tf.float32)
        sn = sn_calc(w)
        return w / sn
