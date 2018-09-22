import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM, CUDNN_RNN_BIDIRECTION
from tensorflow.contrib.cudnn_rnn.python.ops import packing_ops

lengths = [6,3,2]
l = lengths[0]
n = len(lengths)
m = 1
data_dtype = tf.float32
index_dtype = tf.int32

# Create standard CudnnLSTM
lstm = CudnnLSTM(
    num_layers=1, num_units=m, dtype=data_dtype, direction=CUDNN_RNN_BIDIRECTION,
    kernel_initializer=tf.initializers.constant(0.2),
    bias_initializer=tf.initializers.constant(0.2))

# Standard 3d sequence inputs
inputs = tf.ones(shape=(l, n, m), dtype=data_dtype)
# Length of each sequence in decreasing order
sequence_lengths = tf.constant(lengths, dtype=index_dtype)
# Calculate alignments
align = packing_ops.packed_sequence_alignment(sequence_lengths)
# Pack the sequence
packed_inputs = packing_ops.pack_sequence(inputs, *align)
# Run the LSTM
packed_outputs, _ = lstm(packed_inputs, sequence_lengths=sequence_lengths)
# Unpack the outputs
unpacked_outputs = packing_ops.unpack_sequence(packed_outputs, *align)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _sequence_lengths, _packed_outputs, _unpacked_outputs = sess.run([
        sequence_lengths, packed_outputs, unpacked_outputs])
    print("Sequence lengths: {}".format(_sequence_lengths))
    print("Unpacked outputs forward: \n{}".format(_unpacked_outputs[:, :, 0]))
    print("Unpacked outputs backward: \n{}".format(_unpacked_outputs[:, :, 1]))
