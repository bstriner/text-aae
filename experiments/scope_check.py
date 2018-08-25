import tensorflow as tf

from text_aae.sn.sn_conv1d import SNConv1D
from text_aae.sn.sn_linear import SNLinear

with tf.variable_scope('X') as scope:
    with tf.name_scope('X/A/'):
        a1= tf.get_variable('x', dtype=tf.float32, shape=[2], trainable=False)
        b1 = a1+1
        c = SNConv1D(3, 1, dtype=tf.float32, name='conv1d').build((32,30,100))
        d = SNLinear(3, dtype=tf.float32, name='l1').build((32,30,100))

with tf.variable_scope('X', reuse=True):
    with tf.name_scope('X/B/'):
        a2= tf.get_variable('x', dtype=tf.float32, shape=[2], trainable=False)
        b2 = a2+1
        c = SNConv1D(3,1, dtype=tf.float32, name='conv1d').build((32,30,100))
        d = SNLinear(3, dtype=tf.float32, name='l1').build((32,30,100))

print(a1)
print(b1)
print(a2)
print(b2)

print(tf.trainable_variables())