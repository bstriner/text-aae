import tensorflow as tf
import tensorflow.contrib.gan as tfgan

from .gan import GanConfig, Gan
from .images import add_gan_model_image_summaries
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

def model_fn(features, labels, mode, params):
    # Build the generator and discriminator.
    real_data = features['x']
    n = tf.shape(real_data)[0]
    generator_inputs = tf.random_normal(shape=(n, params.latent_units), mean=0., stddev=1., dtype=tf.float32)
    generator_inputs.set_shape((real_data.shape[0], None))
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    gan = Gan(
        xreal=real_data, zsample=generator_inputs,
        params=params, is_training=is_training,
        config=gan_config
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        add_gan_model_image_summaries(
            gan.xreal,
            gan.xfake,
            grid_size=tf.flags.FLAGS.grid_size)

        discriminator_hook = RunTrainOpsHook(
            train_ops=gan.discriminator_op,
            train_steps=params.discriminator_steps)

        return tf.estimator.EstimatorSpec(mode=mode, loss=gan.generator_loss,
                                          train_op=gan.generator_op,
                                          training_hooks=[discriminator_hook])
    else:
        eval_metric_ops = {}
        return tf.estimator.EstimatorSpec(mode=mode, loss=gan.generator_loss,
                                          eval_metric_ops=eval_metric_ops)

