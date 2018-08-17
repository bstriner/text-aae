import tensorflow as tf
from tensorflow.python.framework import common_shapes
from tensorflow.python.layers.base import Layer

from .sn_kernel import sn_kernel


class SNLinear(Layer):
    def __init__(self, num_units, name=None):
        self.num_units = num_units
        super(SNLinear, self).__init__(name=name)

    def build(self, input_shape):
        with tf.variable_scope(self.name):
            print("input_shape: {}".format(input_shape))
            self.kernel = sn_kernel(
                shape=(input_shape[-1], self.num_units),
                scope='kernel'
            )
            self.bias = self.add_variable(
                "bias",
                shape=[self.num_units],
                initializer=tf.initializers.zeros(dtype=self.dtype))
            self.built = True

    def call(self, inputs, **kwargs):
        h = inputs
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            h = tf.tensordot(h, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            shape = inputs.get_shape().as_list()
            output_shape = shape[:-1] + [self.num_units]
            h.set_shape(output_shape)
        else:
            h = tf.matmul(h, self.kernel)
        h = h + self.bias
        return h


from tensorflow.contrib import slim

slim.fully_connected
