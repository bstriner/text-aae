import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION, CudnnLSTM
from tensorflow.contrib.layers import fully_connected


class Decoder(object):
    def __init__(self, vocab_size, params):
        self.lstms = [
            CudnnLSTM(
                num_layers=1,
                num_units=params.decoder_dim,
                direction=CUDNN_RNN_BIDIRECTION,
                name='decoder_lstm_1'),
            CudnnLSTM(
                num_layers=1,
                num_units=params.decoder_dim,
                direction=CUDNN_RNN_BIDIRECTION,
                name='decoder_lstm_2'),
            CudnnLSTM(
                num_layers=1,
                num_units=params.decoder_dim,
                direction=CUDNN_RNN_BIDIRECTION,
                name='decoder_lstm_3')]
        self.vocab_size = vocab_size

    def call(self, z, is_training=True):
        h = z
        for lstm in self.lstms:
            h, _ = lstm(h)
        h = fully_connected(h, self.vocab_size, activation_fn=None)
        y = h
        return y
