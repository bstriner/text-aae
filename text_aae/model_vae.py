import tensorflow as tf
from tensorflow.contrib import slim

from .callbacks.autoencode import AutoencodeCallback
from .callbacks.generate import GenerateCallback
from .rnd import kl_divergence, sample, softplus_rect


def make_model_vae_fn(
        charset,
        encoder_fn,
        decoder_fn,
        model_mode='rnn'):
    vocab_size = len(charset)

    def model_vae_fn(features, labels, mode, params):
        # Build the generator and discriminator.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        x = features['x']
        if model_mode == 'rnn':
            x = tf.transpose(x, (1, 0))

        with tf.variable_scope('Encoder'):
            z_mu, z_logsigma = encoder_fn(x, vocab_size=vocab_size, params=params, is_training=is_training)
            z_sigma = softplus_rect(z_logsigma)
            z = sample(z_mu, z_sigma)
            kl_losses = kl_divergence(z_mu, z_sigma)
        with tf.variable_scope('Decoder') as dec_scope:
            logits = decoder_fn(z, vocab_size=vocab_size, params=params, is_training=is_training)
            pred = tf.argmax(logits, axis=-1)

        with tf.variable_scope(dec_scope, reuse=True):
            prior_z = tf.random_normal(shape=tf.shape(z))
            gen_logits = decoder_fn(prior_z, vocab_size=vocab_size, params=params, is_training=is_training)
            gen_pred = tf.argmax(gen_logits, axis=-1)

        onehot = tf.one_hot(x, axis=-1, depth=vocab_size)
        print("test: {},{}".format(x, logits))

        with tf.name_scope("Losses"):
            sm_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=onehot,
                logits=logits
            )

            # kl_weight = tf.get_variable('kl_weight',shape=[], dtype=tf.float32)
            step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
            anneal_weight = tf.nn.sigmoid((step / params.anneal_rate) - params.anneal_offset)
            tf.summary.scalar("anneal_weight", anneal_weight)

            sm_loss = tf.reduce_mean(sm_losses)
            kl_loss_unscaled = tf.reduce_mean(kl_losses)
            kl_loss = anneal_weight * kl_loss_unscaled

            tf.losses.add_loss(sm_loss)
            tf.losses.add_loss(kl_loss)

        total_loss = tf.losses.get_total_loss()

        if mode == tf.estimator.ModeKeys.TRAIN:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(x, tf.cast(pred, x.dtype)), tf.float32))
            tf.summary.scalar("train_accuracy", accuracy)
            tf.summary.scalar("train_sm_loss", sm_loss)
            tf.summary.scalar("train_kl_loss", kl_loss)
            tf.summary.scalar("train_kl_loss_unscaled", kl_loss_unscaled)
            train_op = slim.learning.create_train_op(
                total_loss=total_loss,
                optimizer=tf.train.AdamOptimizer(learning_rate=params.lr))
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        else:
            autoencode_cb = AutoencodeCallback(
                step=tf.train.get_or_create_global_step(),
                x=x,
                y=pred,
                charset=charset,
                model_mode=model_mode
            )
            generate_cb = GenerateCallback(
                step=tf.train.get_or_create_global_step(),
                gen_text=gen_pred,
                charset=charset,
                model_mode=model_mode
            )
            eval_metric_ops = {
                'val_accuracy': tf.metrics.accuracy(labels=x, predictions=pred),
                'val_sm_loss': tf.metrics.mean(sm_loss),
                'val_kl_loss': tf.metrics.mean(kl_loss)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[autoencode_cb, generate_cb])

    return model_vae_fn
