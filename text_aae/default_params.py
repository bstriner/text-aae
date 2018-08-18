import json
import os

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib.training import HParams


def write_hparams_event(model_dir, hparams):
    mdir = model_dir
    with tf.summary.FileWriter(mdir) as f:
        pjson = tf.constant(hparams_string(hparams), tf.string)
        pten = tf.constant(hparams_array(hparams), tf.string)
        with tf.Session() as sess:
            sstr = tf.summary.text('hparams_json', pjson)
            sten = tf.summary.text('hparams_table', pten)
            sumstr, sumten = sess.run([sstr, sten])
        f.add_summary(sumstr)
        f.add_summary(sumten)


def hparams_string(hparams):
    return json.dumps(hparams.values(), indent=4, sort_keys=True)


def hparams_array(hparams):
    d = hparams.values()
    keys = list(d.keys())
    keys.sort()
    values = [str(d[k]) for k in keys]

    arr = np.array([keys, values], dtype=np.string_).transpose((1, 0))
    return arr


def get_hparams(model_dir, create):
    hparams = default_params()
    hparams_path = os.path.join(model_dir, 'configuration-hparams.json')
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparam_dict = json.load(f)
            for k, v in six.iteritems(hparam_dict):
                setattr(hparams, k, v)
    else:
        if create:
            hparams.parse(tf.flags.FLAGS.hparams)
            with open(hparams_path, 'w') as f:
                json.dump(hparams.values(), f)
            write_hparams_event(model_dir=model_dir, hparams=hparams)
        else:
            raise ValueError("No hparams file found: {}".format(hparams_path))
    return hparams


def default_params():
    return HParams(
        generator_steps=1,
        discriminator_steps=7,
        gen_lr=3e-4,
        dis_lr=3e-4,

        # Network
        encoder_dim=256,
        decoder_dim=256,
        latent_dim=128,
        discriminator_dim=256,
        feature_dim=128,
        lr=3e-4,
        sm_weight=1.,
        gan_weight=1.
    )
