import tensorflow as tf

import text_aae.trainer
from text_aae.model_aae import make_model_aae_fn
from text_aae.networks.cnn.decoder import decoder_cnn_fn
from text_aae.networks.cnn.discriminator import discriminator_cnn_fn
from text_aae.networks.cnn.encoder import encoder_cnn_fn
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
        model_fn=make_model_aae_fn(
            charset=charset,
            encoder_fn=encoder_cnn_fn,
            decoder_fn=decoder_cnn_fn,
            discriminator_fn=discriminator_cnn_fn,
            model_mode=model_mode
        ),
        input_fns=input_fns,
        mode=model_mode
    )
    text_aae.trainer.main(argv, config=config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', 'output/aae/cnn/v21', 'Model directory')
    tf.flags.DEFINE_string('data_dir', 'c:/projects/data/wikitext/wikitext-2', 'Data directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.flags.DEFINE_integer('batch_length', 30, 'Batch length')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('min_after_dequeue', 2000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('grid_size', 10, 'grid_size')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 1000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
