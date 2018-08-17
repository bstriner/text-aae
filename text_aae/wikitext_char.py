import os

import numpy as np
import tensorflow as tf


def readtext(f):
    with open(f, 'r', encoding='utf8') as fi:
        return fi.read()


def make_charset(x):
    l = list(set(x))
    l.sort()
    return l


def map_chars(text, charset):
    charmap = {c: i for i, c in enumerate(charset)}
    return np.array([charmap[c] for c in text if c in charset], dtype=np.int32)


def make_batches(x, batch_length):
    n = x.shape[0]
    batch_count = n // batch_length
    x = x[:batch_count * batch_length]
    return x.reshape((batch_count, batch_length))  # (n, len)


def make_wikitext_char_input_fn(data_dir, batch_length, batch_size):
    files = ['train', 'valid', 'test']
    filenames = {f: 'wiki.{}.tokens'.format(f) for f in files}
    data = {f: readtext(os.path.join(data_dir, filenames[f])) for f in files}
    charset = make_charset(data['train'])
    raw = {f: map_chars(data[f], charset) for f in files}
    batches = {f: make_batches(raw[f], batch_length) for f in files}
    fns = {f: tf.estimator.inputs.numpy_input_fn(
        {'x': batches[f]},
        y=batches[f],
        batch_size=batch_size,
        num_epochs=None,
        shuffle=(f != 'test')
    ) for f in files}
    print("Charset: {}".format(len(charset)))
    return fns, charset
