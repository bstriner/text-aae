import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops.nn_ops import Convolution

from .sn_kernel import sn_kernel


class SNConv1D(Conv1D):
    def build(self, input_shape):
        with tf.variable_scope(self.name):
            input_shape = TensorShape(input_shape)
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
            if input_shape[channel_axis].value is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
            input_dim = int(input_shape[channel_axis])
            kernel_shape = self.kernel_size + (input_dim, self.filters)

            print("input_shape: {}".format(input_shape))
            self.kernel = sn_kernel(
                shape=kernel_shape,
                scope='kernel'
            )
            self.bias = self.add_variable(
                "bias",
                shape=[self.filters],
                initializer=tf.initializers.zeros(dtype=self.dtype))
            self._convolution_op = Convolution(
                input_shape,
                filter_shape=self.kernel.get_shape(),
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self.padding.upper(),
                data_format=conv_utils.convert_data_format(self.data_format,
                                                           self.rank + 2))
            self.built = True
