import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


def generator_train_op(params, scope, loss, opt):
    parameters = tf.trainable_variables(scope.name)
    updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope.name)
    print("Gen updates: {}".format(updates))
    print("Gen parameters: {}".format(parameters))

    tf.summary.scalar('generator_loss', loss)

    train_op = slim.learning.create_train_op(
        total_loss=loss,
        optimizer=opt,
        update_ops=updates,
        variables_to_train=parameters,
        global_step=tf.train.get_or_create_global_step()
    )
    return train_op


def discriminator_train_op(params, scope, loss, opt):
    parameters = tf.trainable_variables(scope.name)
    updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope.name)
    print("Dis updates: {}".format(updates))
    print("Dis parameters: {}".format(parameters))

    tf.summary.scalar('discriminator_loss', loss)

    train_op = slim.learning.create_train_op(
        total_loss=loss,
        optimizer=opt,
        update_ops=updates,
        variables_to_train=parameters,
        global_step=None
    )
    return train_op
