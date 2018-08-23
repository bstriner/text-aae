import csv
import os

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook

from .text import convert_texts
import numpy as np


class AutoencodeCallback(SessionRunHook):
    def __init__(self, step, x, y, charset, model_mode='rnn'):
        self.step = step
        self.x = x
        self.y = y
        self.charset = charset
        self.model_mode=model_mode

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        step, x, y = session.run([self.step, self.x, self.y])
        if not self.model_mode=='rnn':
            x = np.transpose(x, (1,0))
            y = np.transpose(y, (1,0))
        path = os.path.join(tf.flags.FLAGS.model_dir, 'autoencoded', "{:08d}.csv".format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            trues = list(convert_texts(x, charset=self.charset))
            preds = list(convert_texts(y, charset=self.charset))
            w = csv.writer(f)
            w.writerow(['Id', 'True', 'Pred'])
            for i, (t, p) in enumerate(zip(trues, preds)):
                w.writerow([i, t, p])
