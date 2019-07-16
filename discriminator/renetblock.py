import math
import tensorflow as tf
from layers.conv_sn import SNConv2D


class Block(tf.keras.Model):
  def __init__(self, in_channels, out_channels, hidden_channels=None,
               ksize=3, pad="SAME", downsample=False, activation=tf.keras.layers.ReLU()):
    super(Block, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation
    self.downsample = downsample
    self.learnable_sc = (in_channels != out_channels) or downsample
    hidden_channels = in_channels if hidden_channels is None else hidden_channels

    self.c1 = SNConv2D(hidden_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self.c2 = SNConv2D(out_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self._downsample = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="SAME")

    if self.learnable_sc:
      self.c_sc = SNConv2D(out_channels, kernel_size=1, padding="VALID", kernel_initializer=initializer)

  def residual(self, x, **kwargs):
    h = x
    h = self.activation(h)
    h = self.c1(h, **kwargs)
    h = self.activation(h)
    h = self.c2(h, **kwargs)
    if self.downsample:
      h = self._downsample(h)
    return h

  def shortcut(self, x, **kwargs):
    if self.learnable_sc:
      x = self.c_sc(x, **kwargs)
      if self.downsample:
        return self._downsample(x)
      else:
        return x
    else:
      return x

  def __call__(self, x, **kwargs):
    return self.residual(x, **kwargs) + self.shortcut(x, **kwargs)


class OptimizedBlock(tf.keras.Model):
  def __init__(self, out_channels, ksize=3, pad="SAME", activation=tf.keras.layers.ReLU()):
    super(OptimizedBlock, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation

    self.c1 = SNConv2D(out_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self.c2 = SNConv2D(out_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self.c_sc = SNConv2D(out_channels, kernel_size=1, padding="VALID", kernel_initializer=initializer)
    self._downsample = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="VALID")

  def residual(self, x, **kwargs):
    h = x
    h = self.c1(h, **kwargs)
    h = self.activation(h)
    h = self.c2(h, **kwargs)
    h = self._downsample(h)
    return h

  def shortcut(self, x, **kwargs):
    return self.c_sc(self._downsample(x), **kwargs)

  def __call__(self, x, **kwargs):
    return self.residual(x, **kwargs) + self.shortcut(x, **kwargs)
