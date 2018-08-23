import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.python.estimator.estimator import Estimator

from .text_config import TextConfig
from .wikitext_char import make_wikitext_char_input_fn


def make_experiment_fn(config: TextConfig):
    def experiment_fn(run_config, hparams):
        estimator = Estimator(
            model_fn=config.model_fn,
            config=run_config,
            params=hparams)
        experiment = Experiment(
            estimator=estimator,
            train_input_fn=config.input_fns['train'],
            eval_input_fn=config.input_fns['valid']
        )

        return experiment

    return experiment_fn
