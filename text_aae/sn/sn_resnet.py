# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
from tensorflow.contrib import slim

def sn_building_block_v2(inputs, filters, training, projection_shortcut, strides,
                      data_format, activation=tf.nn.relu, kernel_size=3):
    """A single block for ResNet v2, without a bottleneck.
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = slim.batch_norm(inputs, is_training=training, data_format=data_format, scope='resnet_bn_0')
    inputs = activation(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = slim.conv1d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        data_format=data_format, scope='resnet_conv1d_0', activation_fn=None)

    inputs = slim.batch_norm(inputs, is_training=training, data_format=data_format, scope='resnet_bn_1')
    inputs = activation(inputs)
    inputs = slim.conv1d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=1,
        data_format=data_format, scope='resnet_conv1d_1', activation_fn=None)

    return inputs + shortcut


def bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                        strides, data_format, activation=tf.nn.relu, kernel_size=3):
    """A single block for ResNet v2, with a bottleneck.
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of:
      Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = slim.batch_norm(inputs, is_training=training, data_format=data_format, scope='resnet_bn_0')
    inputs = activation(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = slim.conv1d(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format, scope='resnet_conv1d_0', activation_fn=None)

    inputs = slim.batch_norm(inputs, is_training=training, data_format=data_format, scope='resnet_bn_1')
    inputs = activation(inputs)
    inputs = slim.conv1d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        data_format=data_format, scope='resnet_conv1d_1', activation_fn=None)

    inputs = slim.batch_norm(inputs, is_training=training, data_format=data_format, scope='resnet_bn_2')
    inputs = activation(inputs)
    inputs = slim.conv1d(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format, scope='resnet_conv1d_2', activation_fn=None)

    return inputs + shortcut
