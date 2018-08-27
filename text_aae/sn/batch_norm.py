import tensorflow as tf


def interpolate(a, b, alpha):
    return a * (1. - alpha) + b * alpha


def activation_0norm(x):
    norm = tf.reduce_max(tf.abs(x))
    norm = tf.maximum(norm, 1)
    return x / norm


def make_batch_norm(alpha_rate=0.01, beta_rate=0.01, eps=1e-1, stop_grad=False, reg_weight=0.,
                    enable_alpha=True, enable_beta=True, beta_activation=activation_0norm):
    def batch_norm(x, scope, is_training=True):
        with tf.variable_scope(scope):
            dim = x.shape[-1].value
            running_mean = tf.get_variable(
                name='running_mean',
                shape=(dim,),
                dtype=tf.float32,
                initializer=tf.initializers.zeros,
                trainable=False)
            running_var = tf.get_variable(
                name='running_var',
                shape=(dim,),
                dtype=tf.float32,
                initializer=tf.initializers.ones,
                trainable=False)
            reduction_dim = list(range(len(x.shape) - 1))
            actual_mean = tf.reduce_mean(x, axis=reduction_dim)
            actual_var = tf.reduce_mean(tf.square(x - actual_mean), axis=reduction_dim)
            new_mean = interpolate(running_mean, actual_mean, alpha=alpha_rate)
            new_var = interpolate(running_var, actual_var, alpha=beta_rate)
            running_mean = tf.assign(running_mean, new_mean)
            running_var = tf.assign(running_var, new_var)
            tf.add_to_collection(running_mean, tf.GraphKeys.UPDATE_OPS)
            tf.add_to_collection(running_var, tf.GraphKeys.UPDATE_OPS)
            if reg_weight > 0:
                # reg = reg_weight * tf.reduce_sum(1./tf.square(eps+actual_var))
                reg = -reg_weight * tf.reduce_sum(tf.square(actual_var))
                tf.losses.add_loss(reg, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            if stop_grad:
                actual_mean = tf.stop_gradient(actual_mean)
                actual_var = tf.stop_gradient(actual_var)
            if is_training:
                denom = tf.sqrt(actual_var + eps)
                # if clip > 0:
                #    denom = tf.maximum(denom, clip)
                x = (x - actual_mean) / denom
            else:
                denom = tf.sqrt(running_var + eps)
                # if clip > 0:
                #    denom = tf.maximum(denom, clip)
                x = (x - running_mean) / denom
            if enable_beta:
                beta = tf.get_variable(
                    name='beta',
                    shape=(dim,),
                    dtype=tf.float32,
                    initializer=tf.initializers.ones)
                if beta_activation:
                    beta = beta_activation(beta)
                x = x * beta
            if enable_alpha:
                alpha = tf.get_variable(
                    name='alpha',
                    shape=(dim,),
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros)
                x = x + alpha
            return x

    return batch_norm
