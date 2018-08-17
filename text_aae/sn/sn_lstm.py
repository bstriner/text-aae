import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.rnn_cell import LSTMCell

from .sn_kernel import sn_kernel


class SNLSTMCell(LSTMCell):
    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        print("SNLSTMCell inputs_shape: {}".format(inputs_shape))
        with tf.variable_scope(self.name):
            if inputs_shape[-1] is None:
                raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                 % inputs_shape)

            input_depth = inputs_shape[-1]
            h_depth = self._num_units if self._num_proj is None else self._num_proj

            self._kernels = [
                sn_kernel(
                    shape=(input_depth + h_depth, self._num_units),
                    scope='kernel_{}'.format(i)
                )
                for i in range(4)
            ]
            # for k in self._kernels:
            #    self._trainable_weights.append(k)
            self._kernel = tf.concat(self._kernels, axis=1)

            initializer = tf.initializers.zeros()
            self._bias = self.add_variable(
                "bias",
                shape=[4 * self._num_units],
                initializer=initializer)
            assert not self._use_peepholes

            if self._num_proj is not None:
                self._proj_kernel = sn_kernel(
                    scope="projection/kernel",
                    shape=[self._num_units, self._num_proj]
                )
            self.built = True
