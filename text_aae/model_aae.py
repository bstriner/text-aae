import tensorflow as tf
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .callbacks.autoencode import AutoencodeCallback
from .callbacks.generate import GenerateCallback
from .gan.losses import gan_losses
from .gan.train import discriminator_train_op, generator_train_op
from .rnd import sample, softplus_rect


def make_model_aae_fn(
        charset,
        encoder_fn,
        decoder_fn,
        discriminator_fn,
        model_mode='rnn'):
    vocab_size = len(charset)

    def model_aae_fn(features, labels, mode, params):
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
                z = sample(z_mu, z_sigma)
            with tf.variable_scope("Decoder"):
                logits = decoder_fn(
                    z,
                    vocab_size=vocab_size,
                    params=params,
                    is_training=is_training)

        if True:
            with tf.variable_scope("Discriminator") as dis_scope:
                z_prior = tf.random_normal(shape=tf.shape(z))
                all_z = tf.concat([z_prior, z], axis=batch_dim)
                all_y, all_h = discriminator_fn(
                    all_z, params=params, is_training=is_training)
                n = x.shape[batch_dim].value
                y_real = all_y[:n, :]
                y_fake = all_y[n:, :]
                h_real = all_h[:n, :]
                h_fake = all_h[n:, :]
        else:
            with tf.variable_scope("Discriminator") as dis_scope:
                z_prior = tf.random_normal(shape=tf.shape(z))
                with tf.name_scope("Discriminator/Real/"):
                    y_real, h_real = discriminator_fn(
                        z_prior, params=params, is_training=is_training)
            with tf.variable_scope(dis_scope, reuse=True):
                with tf.name_scope("Discriminator/Fake/"):
                    y_fake, h_fake = discriminator_fn(
                        z, params=params, is_training=is_training)

        gloss, dloss = gan_losses(y_real=y_real, y_fake=y_fake, h_real=h_real, h_fake=h_fake)

        pred = tf.argmax(logits, axis=-1)
        onehot = tf.one_hot(x, axis=-1, depth=vocab_size)

        with tf.variable_scope(aae_scope, reuse=True):
            with tf.name_scope('Autoencoder/Losses/'):
                sm_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=onehot,
                    logits=logits
                )
                sm_loss = tf.reduce_mean(sm_losses) * params.sm_weight
                gloss = gloss * params.gan_weight
                # tf.losses.add_loss(sm_loss)
                # tf.losses.add_loss(gloss)
                gen_loss = gloss + sm_loss
                # gen_loss = sm_loss
            with tf.variable_scope("Decoder"):
                with tf.name_scope('Generation/'):
                    gen_logits = decoder_fn(
                        z_prior,
                        vocab_size=vocab_size,
                        params=params,
                        is_training=is_training)
                    gen_text = tf.argmax(gen_logits, axis=-1)

        # with tf.variable_scope(dis_scope):
        #    tf.losses.add_loss(dloss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(x, tf.cast(pred, x.dtype)), tf.float32))
            tf.summary.scalar("train_accuracy", accuracy)
            tf.summary.scalar("gen_loss_gan", gloss)
            tf.summary.scalar("gen_loss_sm", sm_loss)
            tf.summary.scalar("gen_loss_total", gen_loss)
            tf.summary.scalar("dis_loss", dloss)

            dis_train_op = discriminator_train_op(params=params, scope=dis_scope, loss=dloss)
            gen_train_op = generator_train_op(params=params, scope=aae_scope, loss=gen_loss)

            discriminator_hook = RunTrainOpsHook(
                train_ops=dis_train_op,
                train_steps=params.discriminator_steps)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=gen_loss,
                train_op=gen_train_op,
                training_hooks=[discriminator_hook])
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
                gen_text=gen_text,
                charset=charset,
                model_mode=model_mode
            )
            eval_metric_ops = {
                'val_accuracy': tf.metrics.accuracy(labels=x, predictions=pred)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=gen_loss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[autoencode_cb, generate_cb])

    return model_aae_fn
