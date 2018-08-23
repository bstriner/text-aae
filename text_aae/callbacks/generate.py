import csv
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook

from .text import convert_texts


class GenerateCallback(SessionRunHook):
    def __init__(self, step, gen_text, charset, model_mode='rnn'):
        self.step = step
        self.gen_text = gen_text
        self.charset = charset
        self.model_mode = model_mode

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        step, gen_text = session.run([self.step, self.gen_text])
        path = os.path.join(tf.flags.FLAGS.model_dir, 'generated', "{:08d}.csv".format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.model_mode == 'rnn':
            gen_text = np.transpose(gen_text, (1, 0))
        with open(path, 'w', newline='', encoding='utf-8') as f:
            gens = list(convert_texts(gen_text, charset=self.charset))
            w = csv.writer(f)
            w.writerow(['Id', 'Generated'])
            for i, g in enumerate(gens):
                w.writerow([i, g])
