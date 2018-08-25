import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense

from .sn_kernel import sn_kernel


class SNLinear(Dense):
    def build(self, input_shape):
        with tf.variable_scope(self.name):
            self.kernel = sn_kernel(
                shape=(input_shape[-1], self.units),
                scope='kernel'
            )
            self.bias = tf.get_variable(
                name='bias',
                shape=[self.units],
                initializer=tf.initializers.zeros(dtype=self.dtype)
            )
            self.built = True
