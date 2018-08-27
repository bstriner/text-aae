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


def make_text_input_fn(text, batch_length, batch_size):
    n = text.shape[0]
    maxn = n - batch_length + 1

    def input_fn():
        ctext = tf.constant(text, name='input_text', dtype=tf.int32)
        start_idx = tf.random_uniform(minval=0, maxval=maxn, dtype=tf.int32, shape=(batch_size, 1))
        arange = tf.range(start=0, limit=batch_length, dtype=tf.int32)
        idx = start_idx + arange
        print("input: {}, {}".format(idx, ctext))

        #batch = tf.reshape(ctext[tf.reshape(idx,(-1,))],tf.shape(idx))
        batch = tf.gather(params=ctext, indices=idx, axis=0)
        return {'x': batch}, None

    return input_fn


def make_wikitext_char_input_fn(data_dir, batch_length, batch_size):
    files = ['train', 'valid', 'test']
    filenames = {f: 'wiki.{}.tokens'.format(f) for f in files}
    data = {f: readtext(os.path.join(data_dir, filenames[f])) for f in files}
    charset = make_charset(data['train'])
    raw = {f: map_chars(data[f], charset) for f in files}
    fns = {f: make_text_input_fn(raw[f], batch_length=batch_length, batch_size=batch_size) for f in files}
    print("Charset: {}".format(len(charset)))
    return fns, charset
