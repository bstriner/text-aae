import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.python.estimator.estimator import Estimator

from .text_config import TextConfig
from .wikitext_char import make_wikitext_char_input_fn


def make_experiment_fn(config: TextConfig):
    def experiment_fn(run_config, hparams):
        input_fns, charset = make_wikitext_char_input_fn(
            data_dir=tf.flags.FLAGS.data_dir,
            batch_length=tf.flags.FLAGS.batch_length,
            batch_size=tf.flags.FLAGS.batch_size
        )
        estimator = Estimator(
            model_fn=config.make_model_fn(charset),
            config=run_config,
            params=hparams)
        experiment = Experiment(
            estimator=estimator,
            train_input_fn=input_fns['train'],
            eval_input_fn=input_fns['valid']
        )

        return experiment

    return experiment_fn
