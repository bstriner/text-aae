import tensorflow as tf
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .callbacks.autoencode import AutoencodeCallback
from .callbacks.generate import GenerateCallback
from text_aae.networks.rnn.decoder import Decoder
from text_aae.networks.rnn.discriminator import Discriminator
from text_aae.networks.rnn.encoder import Encoder
from .gan.losses import gan_losses_h
from .gan.train import discriminator_train_op, generator_train_op
from .text_config import TextConfig

def make_model_aae_fn(charset, config:TextConfig):
    vocab_size = len(charset)

    def model_aae_fn(features, labels, mode, params):
        # Build the generator and discriminator.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        x = features['x']
        x = tf.transpose(x, (1, 0))

        with tf.variable_scope("Autoencoder") as aae_scope:
            with tf.variable_scope("Encoder"):
                enc = Encoder(vocab_size=vocab_size, params=params)
                z = enc.call(x, is_training=is_training)
            with tf.variable_scope("Decoder"):
                dec = Decoder(vocab_size=vocab_size, params=params)
                logits = dec.call(z, is_training=is_training)

        with tf.variable_scope("Discriminator") as dis_scope:
            dis = Discriminator(params=params)
            z_prior = tf.random_normal(shape=tf.shape(z))
            all_z = tf.concat([z_prior, z], axis=1)
            all_y = dis.call(all_z, is_training=is_training)
            n = x.shape[1].value
            y_real = all_y[:n, :]
            y_fake = all_y[n:, :]
            #y_real = dis.call(z_prior, is_training=is_training)
            #y_fake = dis.call(z, is_training=is_training)

        gloss, dloss = gan_losses_h(y_real=y_real, y_fake=y_fake)

        pred = tf.argmax(logits, axis=-1)
        onehot = tf.one_hot(x, axis=-1, depth=vocab_size)

        with tf.variable_scope(aae_scope):
            sm_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=onehot,
                logits=logits
            )
            sm_loss = tf.reduce_mean(sm_losses) * params.sm_weight
            gloss = gloss * params.gan_weight
            tf.losses.add_loss(sm_loss)
            tf.losses.add_loss(gloss)
            gen_loss = gloss + sm_loss
            #gen_loss = sm_loss
            gen_logits = dec.call(z_prior)
            gen_text = tf.argmax(gen_logits, axis=-1)

        with tf.variable_scope(dis_scope):
            tf.losses.add_loss(dloss)

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
                charset=charset
            )
            generate_cb = GenerateCallback(
                step=tf.train.get_or_create_global_step(),
                gen_text=gen_text,
                charset=charset
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
