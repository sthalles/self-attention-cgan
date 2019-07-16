import tensorflow as tf
from layers.ops import power_iteration


class SNTransposeConv2D(tf.keras.Model):
  def __init__(self, filters, kernel_size=3, strides=1, padding='SAME', max_iters=1,
               kernel_initializer='normal', use_bias=True):
    super(SNTransposeConv2D, self).__init__()
    self.max_iters = max_iters
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias
    self.filters = filters
    self.kernel_initializer=kernel_initializer
    self.kernel_size=kernel_size
    self.u = None

  def compute_spectral_normal(self, weights, training):
    # Spectrally Normalized Weight
    W_shape = weights.shape.as_list()

    if self.u is None:
      # =[number_classes, embedding_size]
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

  def build(self, input_shape):
    # kernel shape [kernel_size, kernel_size, output_channels, in_channels]
    self.kernel = self.add_weight(shape=(self.kernel_size, self.kernel_size, self.filters, input_shape[-1]),
                                  initializer=self.kernel_initializer,
                                  trainable=True,
                                  name='kernel')
    if self.use_bias:
      self.biases = self.add_weight(name='biases', shape=[self.filters],
                                  initializer='zeros', trainable=True)


  def call(self, x, training):
    shape = x.shape
    output_shape = [shape[0], shape[1]*2, shape[2]*2, self.filters]

    w_bar = self.compute_spectral_normal(self.kernel, training)
    deconv = tf.nn.conv2d_transpose(x, w_bar, output_shape=output_shape,  strides=[1, self.strides, self.strides, 1], padding=self.padding)

    if self.use_bias:
      deconv= tf.nn.bias_add(deconv, self.biases)

    deconv.shape.assert_is_compatible_with(output_shape)
    return deconv