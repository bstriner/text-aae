import tensorflow as tf

from text_aae.sn.sn_linear import SNLinear
from text_aae.sn.sn_lstm import SNLSTMCell
from tensorflow.python.ops.rnn_cell import LSTMCell


class Discriminator(object):
    def __init__(self, params):
        self.lstm_cells_fw = [
            SNLSTMCell(
                num_units=params.decoder_dim,
                name='discriminator_lstm_1_fw'),
            SNLSTMCell(
                num_units=params.discriminator_dim,
                name='discriminator_lstm_2_fw'),
            SNLSTMCell(
                num_units=params.discriminator_dim,
                name='discriminator_lstm_3_fw')
        ]
        self.lstm_cells_bw = [
            SNLSTMCell(
                num_units=params.decoder_dim,
                name='discriminator_lstm_1_bw'),
            SNLSTMCell(
                num_units=params.discriminator_dim,
                name='discriminator_lstm_2_bw'),
            SNLSTMCell(
                num_units=params.discriminator_dim,
                name='discriminator_lstm_3_bw')
        ]
        self.projection = SNLinear(
            num_units=1 #params.feature_dim
        )

    def call(self, z, is_training=True):
        h = z
        for i, (cell_fw, cell_bw) in enumerate(zip(self.lstm_cells_fw, self.lstm_cells_bw)):
            print("Hdis: {}".format(h))
            cell_fw.build((h.shape[-2].value, h.shape[-1].value))
            cell_bw.build((h.shape[-2].value, h.shape[-1].value))
            (hfw, hbw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=h,
                time_major=True,
                dtype=tf.float32)
            print("hfw hbw: {}, {}".format(hfw, hbw))
            h = tf.concat((hfw, hbw), axis=-1, name='fw_bw_concat_{}'.format(i))

        h = tf.reduce_mean(h, axis=0)
        h = self.projection(h)
        y = h
        return y
