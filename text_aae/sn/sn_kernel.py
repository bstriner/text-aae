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
    uup = tf.assign(u, ut, name='update_u')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, uup)
    sn = spec_norm(uup, w)
    if len(shape)>2:
        scale = np.sqrt(np.prod(shape[:-2]))
    else:
        scale = 1.
    sn = tf.maximum(sn*scale, 1.)
    return sn


def sn_kernel(shape, scope, init=tf.initializers.glorot_normal()):
    with tf.variable_scope(scope) as vs:
        kernel_name = vs.name + "/kernel_sn:0"
        try:
            kernel = tf.get_default_graph().get_tensor_by_name(kernel_name)
            print("Existing kernel: {}".format(kernel_name))
            return kernel
        except KeyError:
            with tf.name_scope(vs.name + "/"):
                w = tf.get_variable(name='kernel_raw', shape=shape, initializer=init, dtype=tf.float32)
                sn = sn_calc(w)
                if len(shape)>2:
                    #scale = tf.reduce_prod(tf.cast(tf.shape(w)[:-2], tf.float32))
                    #scale = tf.sqrt(tf.reduce_prod(tf.cast(tf.shape(w)[:-2], tf.float32)))
                    scale = 1.
                else:
                    scale = 1.
                kernel = tf.div(w, sn*scale, name='kernel_sn')
                s, u, v = tf.linalg.svd(tf.reshape(kernel, (-1, tf.shape(kernel)[-1])))
                tf.summary.scalar(vs.name + "/max_sv", tf.reduce_max(tf.abs(s)))
                print("New kernel: {}".format(kernel_name))
                return kernel
