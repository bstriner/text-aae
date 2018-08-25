import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from text_aae.sn.sn_conv1d import SNConv1D
from text_aae.sn.sn_linear import SNLinear

n = 32
l = 500
m = 230
epochs = 1000
dim = 128
opt = tf.train.AdamOptimizer(1e-3)
layers = 6

a = np.zeros((n, l, m), dtype=np.float32)
b = np.zeros((n, l, m), dtype=np.float32)
a[:, :, 0] = 1


# b[:, :, 1] = 1

def diffs(ha, hb):
    ds = []
    for a, b in zip(ha, hb):
        ds.append(tf.reduce_mean(tf.norm(a - b, ord=2, axis=-1)))
    return ds


def dfn(x):
    # vs = tf.get_variable_scope()
    hiddens = []
    h = x
    hiddens.append(h)
    h = SNLinear(dim, name='l1')(h)
    hiddens.append(h)
    for i in range(layers):
        h = SNConv1D(
            dim,
            kernel_size=7,
            padding='same',
            activation=tf.nn.leaky_relu,
            name='conv_{}'.format(i))(h)
        hiddens.append(h)
    h = tf.reduce_mean(h, axis=1)
    h = SNLinear(1, name='lout')(h)
    hiddens.append(h)
    return h, hiddens


a = tf.constant(a, name='a')
b = tf.constant(b, name='b')

with tf.variable_scope('DFN') as scope:
    with tf.name_scope('DFN/A/'):
        ya, ah = dfn(a)
with tf.variable_scope(scope, reuse=True):
    with tf.name_scope('DFN/B/'):
        yb, bh = dfn(b)
loss = tf.reduce_mean(yb) - tf.reduce_mean(ya)

updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
parameters = tf.trainable_variables()

print("updates: {}".format(updates))
print("parameters: {}".format(parameters))
assert len(parameters) == 2 * (layers + 2)
train_op = slim.learning.create_train_op(
    total_loss=loss,
    optimizer=opt,
    global_step=tf.train.get_or_create_global_step(),
    update_ops=updates,
    variables_to_train=parameters
)

ds = diffs(ah, bh)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        train_loss, _ = sess.run([loss, train_op])
        print("Loss {}: {}".format(i, train_loss))
    print(sess.run(ds))
