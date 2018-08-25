import tensorflow as tf

import text_aae.trainer
from text_aae.gan.losses import wgan_losses
from text_aae.model_gan import make_model_gan_fn
from text_aae.networks.cnn.decoder import decoder_cnn_fn
from text_aae.networks.cnn.discriminator_gan import discriminator_gan_cnn_fn
from text_aae.text_config import TextConfig
from text_aae.wikitext_char import make_wikitext_char_input_fn


def main(argv):
    input_fns, charset = make_wikitext_char_input_fn(
        data_dir=tf.flags.FLAGS.data_dir,
        batch_length=tf.flags.FLAGS.batch_length,
        batch_size=tf.flags.FLAGS.batch_size
    )
    model_mode = 'cnn'
    config = TextConfig(
        model_fn=make_model_gan_fn(
            charset=charset,
            decoder_fn=decoder_cnn_fn,
            gan_discriminator_fn=discriminator_gan_cnn_fn,
            gan_loss_fn=wgan_losses,
            model_mode=model_mode,
            dis_opt=tf.train.AdamOptimizer(1e-5, name='dis_opt'),
            gen_opt=tf.train.AdamOptimizer(1e-5, name='gen_opt')
        ),
        input_fns=input_fns,
        mode=model_mode
    )
    text_aae.trainer.main(argv, config=config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', 'output/gan/cnn/v22', 'Model directory')
    tf.flags.DEFINE_string('data_dir', 'c:/projects/data/wikitext/wikitext-2', 'Data directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
    tf.flags.DEFINE_integer('batch_length', 40, 'Batch length')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('min_after_dequeue', 2000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('grid_size', 10, 'grid_size')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 1000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()