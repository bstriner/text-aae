import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION, CudnnLSTM
from tensorflow.contrib.layers import fully_connected


class Encoder(object):
    def __init__(self, vocab_size, params):
        self.word_embeddings = tf.get_variable(
            name="encoder_embeddings",
            shape=[vocab_size, params.encoder_dim],
            dtype=tf.float32,
            initializer=tf.initializers.glorot_normal())
        self.lstms = [
            CudnnLSTM(
                num_layers=1,
                num_units=params.encoder_dim,
                direction=CUDNN_RNN_BIDIRECTION,
                name='encoder_lstm_1'),
            CudnnLSTM(
                num_layers=1,
                num_units=params.encoder_dim,
                direction=CUDNN_RNN_BIDIRECTION,
                name='encoder_lstm_2'),
            CudnnLSTM(
                num_layers=1,
                num_units=params.encoder_dim,
                direction=CUDNN_RNN_BIDIRECTION,
                name='encoder_lstm_3')]
        self.latent_dim = params.latent_dim

    def call(self, x, is_training=True):
        h = x
        h = tf.nn.embedding_lookup(self.word_embeddings, h)
        for i, lstm in enumerate(self.lstms):
            print("H: {}".format(h))
            h, s = lstm(h)
            print(s)

        print("H: {}".format(h))
        mu = fully_connected(h, self.latent_dim, activation_fn=None, scope='mu')
        logsigma = fully_connected(h, self.latent_dim, activation_fn=None, scope='logsigma')
        rnd = tf.random_normal(shape=tf.shape(logsigma))
        z = mu + (rnd * tf.exp(logsigma))
        return z
