import tensorflow as tf
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .callbacks.generate import GenerateCallback
from .gan.train import discriminator_train_op, generator_train_op
from .gumbel import gumbel_softmax


def make_model_gan_fn(
        charset,
        decoder_fn,
        gan_discriminator_fn,
        gan_loss_fn,
        gen_opt,
        dis_opt,
        padding_size=0,
        model_mode='rnn',
        combined=True):
    vocab_size = len(charset)

    def model_gan_fn(features, labels, mode, params):
        # Build the generator and discriminator.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        x = features['x']
        if model_mode == 'rnn':
            x = tf.transpose(x, (1, 0))
            batch_dim = 1
            time_dim = 0
        else:
            batch_dim = 0
            time_dim = 1

        n = x.shape[batch_dim].value
        l = x.shape[1 - batch_dim].value
        with tf.variable_scope("Generator") as aae_scope:
            with tf.variable_scope("Decoder"):
                z_prior = tf.random_normal(shape=(n, l+padding_size, params.latent_dim), name='z_prior')
                logits = decoder_fn(
                    z_prior,
                    vocab_size=vocab_size,
                    params=params,
                    is_training=is_training)
                x_model = gumbel_softmax(
                    logits=logits,
                    temperature=0.5,
                    hard=True,
                    axis=-1
                )

        onehot_x = tf.one_hot(x, depth=vocab_size, axis=-1)
        if combined:
            with tf.variable_scope('Discriminator') as dis_scope:
                x_all = tf.concat((onehot_x, x_model), axis=batch_dim)
                y_all, h_all = gan_discriminator_fn(
                    x=x_all,
                    params=params,
                    is_training=is_training)
                if model_mode == 'rnn':
                    y_real, y_fake = y_all[:, :n], y_all[:, n:]
                    h_real, h_fake = h_all[:, :n], h_all[:, n:]
                else:
                    y_real, y_fake = y_all[:n], y_all[n:]
                    h_real, h_fake = h_all[:n], h_all[n:]
        else:
            with tf.variable_scope('Discriminator') as dis_scope:
                with tf.name_scope('Discriminator/Real/'):
                    y_real, h_real = gan_discriminator_fn(
                        x=onehot_x,
                        params=params,
                        is_training=is_training)
            with tf.variable_scope(dis_scope, reuse=True):
                with tf.name_scope('Discriminator/Fake/'):
                    y_fake, h_fake = gan_discriminator_fn(
                        x=x_model,
                        params=params,
                        is_training=is_training)

        gloss, dloss = gan_loss_fn(y_real=y_real, y_fake=y_fake, h_real=h_real, h_fake=h_fake)

        generated_x = tf.argmax(logits, axis=-1)

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("gen_loss_gan", gloss)
            tf.summary.scalar("dis_loss_gan", dloss)

            dis_train_op = discriminator_train_op(params=params, scope=dis_scope, loss=dloss, opt=dis_opt)
            gen_train_op = generator_train_op(params=params, scope=aae_scope, loss=gloss, opt=gen_opt)

            discriminator_hook = RunTrainOpsHook(
                train_ops=dis_train_op,
                train_steps=params.discriminator_steps)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=gloss,
                train_op=gen_train_op,
                training_hooks=[discriminator_hook])
        else:
            generate_cb = GenerateCallback(
                step=tf.train.get_or_create_global_step(),
                gen_text=generated_x,
                charset=charset,
                model_mode=model_mode
            )
            eval_metric_ops = {
                'val_loss_gan': tf.metrics.mean(gloss),
                'val_loss_dis': tf.metrics.mean(dloss)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=gloss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[generate_cb])

    return model_gan_fn
