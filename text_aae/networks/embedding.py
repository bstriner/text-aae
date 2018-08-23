import tensorflow as tf


def embedding_fn(x, vocab_size, dim_out, name='embeddings'):
    word_embeddings = tf.get_variable(
        name=name,
        shape=[vocab_size, dim_out],
        dtype=tf.float32,
        initializer=tf.initializers.random_normal(stddev=0.05))
    y = tf.nn.embedding_lookup(word_embeddings, x)
    return y
