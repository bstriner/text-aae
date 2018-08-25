import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from text_aae.sn.sn_conv1d import SNConv1D
from text_aae.sn.sn_linear import SNLinear

n = 32
l = 20
m = 230

dim = 128

a = np.zeros((n, l, m), dtype=np.float32)
b = np.zeros((n, l, m), dtype=np.float32)
a[:, :, 0] = 1
b[:, :, 1] = 1


def dfn(x):
    #vs = tf.get_variable_scope()
    h = x
    h = SNLinear(dim, name='l1')(h)
    for i in range(6):
        h = SNConv1D(dim, kernel_size=7, padding='same', name='conv_{}'.format(i))(h)
    h = tf.reduce_mean(h, axis=1)
    h = SNLinear(1, name='lout')(h)
    return h


a = tf.constant(a, name='a')
b = tf.constant(b, name='b')

with tf.variable_scope('DFN') as scope:
    with tf.name_scope('DFN/A/'):
        ya = dfn(a)
with tf.variable_scope(scope, reuse=True):
    with tf.name_scope('DFN/B/'):
        yb = dfn(b)
loss = tf.reduce_mean(yb) - tf.reduce_mean(ya)

opt = tf.train.AdamOptimizer(1e-3)

updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
parameters = tf.trainable_variables()


print("updates: {}".format(updates))
print("parameters: {}".format(parameters))
assert len(parameters) == 2*8
train_op = slim.learning.create_train_op(
    total_loss=loss,
    optimizer=opt,
    global_step=tf.train.get_or_create_global_step(),
    update_ops=updates,
    variables_to_train=parameters
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        train_loss, _ = sess.run([loss, train_op])
        print("Loss {}: {}".format(i, train_loss))
