import tensorflow as tf
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .callbacks.autoencode import AutoencodeCallback
from .callbacks.generate import GenerateCallback
from .gan.train import discriminator_train_op, generator_train_op
from .gumbel import gumbel_softmax
from .rnd import sample, softplus_rect


def make_model_bigan_fn(
        charset,
        encoder_fn,
        decoder_fn,
        bidiscriminator_fn,
        gan_loss_fn,
        model_mode='rnn'):
    vocab_size = len(charset)

    def model_bigan_fn(features, labels, mode, params):
        # Build the generator and discriminator.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        x = features['x']
        if model_mode == 'rnn':
            x = tf.transpose(x, (1, 0))
            batch_dim = 1
        else:
            batch_dim = 0

        with tf.variable_scope("Autoencoder") as aae_scope:
            with tf.variable_scope("Encoder"):
                z_mu, z_logsigma = encoder_fn(
                    x,
                    vocab_size=vocab_size,
                    params=params,
                    is_training=is_training)
                z_sigma = softplus_rect(z_logsigma)
                z_model = sample(z_mu, z_sigma)
            with tf.variable_scope("Decoder"):
                z_prior = tf.random_normal(shape=tf.shape(z_model))
                logits = decoder_fn(
                    z_prior,
                    vocab_size=vocab_size,
                    params=params,
                    is_training=is_training)
                x_model = gumbel_softmax(
                    logits=logits,
                    temperature=0.5,
                    hard=True
                )

        with tf.variable_scope('Discriminator') as dis_scope:
            with tf.name_scope('Discriminator/Encoded/'):
                onehot_x = tf.one_hot(x, depth=vocab_size)
                y_real, h_real = bidiscriminator_fn(
                    x=onehot_x,
                    z=z_model,
                    params=params,
                    is_training=is_training)
        with tf.variable_scope(dis_scope, reuse=True):
            with tf.name_scope('Discriminator/Decoded/'):
                y_fake, h_fake = bidiscriminator_fn(
                    x=x_model,
                    z=z_prior,
                    params=params,
                    is_training=is_training)

        gloss, dloss = gan_loss_fn(y_real=y_real, y_fake=y_fake, h_real=h_real, h_fake=h_fake)

        generated_x = tf.argmax(logits, axis=-1)
        with tf.variable_scope(aae_scope, reuse=True):
            with tf.variable_scope('Decoder'):
                autoencoded_logits = decoder_fn(
                    z_model,
                    vocab_size=vocab_size,
                    params=params,
                    is_training=is_training)
                autoencoded_x = tf.argmax(autoencoded_logits, axis=-1)

        if mode == tf.estimator.ModeKeys.TRAIN:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(x, tf.cast(autoencoded_x, x.dtype)), tf.float32))
            tf.summary.scalar("train_accuracy", accuracy)
            tf.summary.scalar("gen_loss_bigan", gloss)
            tf.summary.scalar("dis_loss_bigan", dloss)

            dis_train_op = discriminator_train_op(params=params, scope=dis_scope, loss=dloss)
            gen_train_op = generator_train_op(params=params, scope=aae_scope, loss=gloss)

            discriminator_hook = RunTrainOpsHook(
                train_ops=dis_train_op,
                train_steps=params.discriminator_steps)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=gloss,
                train_op=gen_train_op,
                training_hooks=[discriminator_hook])
        else:
            autoencode_cb = AutoencodeCallback(
                step=tf.train.get_or_create_global_step(),
                x=x,
                y=autoencoded_x,
                charset=charset,
                model_mode=model_mode
            )
            generate_cb = GenerateCallback(
                step=tf.train.get_or_create_global_step(),
                gen_text=generated_x,
                charset=charset,
                model_mode=model_mode
            )
            eval_metric_ops = {
                'val_accuracy': tf.metrics.accuracy(labels=x, predictions=generated_x)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=gloss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[autoencode_cb, generate_cb])

    return model_bigan_fn
