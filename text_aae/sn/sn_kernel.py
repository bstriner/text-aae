import numpy as np
import tensorflow as tf
from tensorflow.python.ops.variable_scope import _get_default_variable_store


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


def sn_calc(w, update=True):
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
    sn = spec_norm(ut, w)
    sn = tf.maximum(sn, 1.)

    if update:
        uup = tf.assign(u, ut, name='update_u')
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, uup)

    return sn


def sn_kernel(shape, scope, init=tf.initializers.glorot_normal()):
    store = _get_default_variable_store()
    with tf.variable_scope(scope) as vs:
        kernel_name = vs.name + "/kernel_raw"
        update = kernel_name not in store._vars
        w = tf.get_variable(name='kernel_raw', shape=shape, initializer=init, dtype=tf.float32)
        sn = sn_calc(w, update=update)
        return tf.div(w, sn, name='kernel_sn')
