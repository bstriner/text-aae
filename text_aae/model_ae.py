import tensorflow as tf
from tensorflow.contrib import slim

from text_aae.networks.rnn.decoder import Decoder
from text_aae.networks.rnn.encoder import Encoder
from .callbacks.autoencode import AutoencodeCallback
from .text_config import TextConfig


def make_model_ae_fn(
        charset,
        encoder_fn,
        decoder_fn,
        model_mode='rnn'):
    vocab_size = len(charset)

    def model_ae_fn(features, labels, mode, params):
        # Build the generator and discriminator.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        x = features['x']
        if model_mode=='rnn':
            x = tf.transpose(x, (1, 0))

        z = encoder_fn(x, vocab_size=vocab_size, params=params, is_training=is_training)
        logits = decoder_fn(z, vocab_size=vocab_size, params=params, is_training=is_training)

        pred = tf.argmax(logits, axis=-1)
        onehot = tf.one_hot(x, axis=-1, depth=vocab_size)
        print("test: {},{}".format(x, logits))

        sm_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=onehot,
            logits=logits
        )
        tf.losses.add_loss(tf.reduce_mean(sm_losses))
        total_loss = tf.losses.get_total_loss()

        if mode == tf.estimator.ModeKeys.TRAIN:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(x, tf.cast(pred, x.dtype)), tf.float32))
            tf.summary.scalar("train_accuracy", accuracy)
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
            eval_metric_ops = {
                'val_accuracy': tf.metrics.accuracy(labels=x, predictions=pred)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[autoencode_cb])

    return model_ae_fn
