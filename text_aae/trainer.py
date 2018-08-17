import os

import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.learn.python.learn.learn_runner import run

from .default_params import get_hparams
from .experiment import make_experiment_fn
from .text_config import TextConfig


def main(_argv, config: TextConfig):
    model_dir = tf.flags.FLAGS.model_dir
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir={}".format(model_dir))
    run_config = RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=tf.flags.FLAGS.save_checkpoints_steps)
    # save_checkpoints_secs=tf.flags.FLAGS.save_checkpoints_secs)
    hparams = get_hparams(model_dir, create=True)
    estimator = run(
        experiment_fn=make_experiment_fn(config),
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)
