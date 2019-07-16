import tensorflow as tf
from layers.ops import power_iteration

class SNEmbeeding(tf.keras.Model):
  def __init__(self, embedding_size, n_classes, kernel_initializer='normal', max_iters=1):
    super(SNEmbeeding, self).__init__()
    self.max_iters = max_iters
    self.embedding_map = self.add_weight(shape=(n_classes, embedding_size),
                                         initializer=kernel_initializer,
                                         trainable=True,
                                         name='embedding_map')
    self.u = None

  def compute_spectral_normal(self, weights, training):
    # Spectrally Normalized Weight
    W_shape = weights.shape.as_list()

    if self.u is None:
      self.u = self.add_weight(
        'sn_estimate',
        shape=[1, W_shape[-1]],
        initializer='normal',
        dtype=weights.dtype,
        trainable=False)

    # Flatten the Tensor
    W_mat = tf.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]
    W_sn, u, v = power_iteration(W_mat, self.u, rounds=self.max_iters)

    if training == True:
      # Update estimated 1st singular vector
      self.u.assign(u)

    W_mat = W_mat / W_sn
    w_bar = tf.reshape(W_mat, W_shape)

    return w_bar

  def call(self, x, training):
    embedding_map_bar_transpose = self.compute_spectral_normal(tf.transpose(self.embedding_map), training=training)
    embedding_map_bar = tf.transpose(embedding_map_bar_transpose)
    return tf.nn.embedding_lookup(embedding_map_bar, x)