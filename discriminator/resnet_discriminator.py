import tensorflow as tf
from discriminator.renetblock import OptimizedBlock, Block
from layers.embedding_sn import SNEmbeeding
from layers.dense_sn import SNDense

class ResnetDiscriminator(tf.keras.Model):
  def __init__(self, ch, activation=tf.keras.layers.ReLU(), n_classes=0):
    super(ResnetDiscriminator, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation
    self.concat = tf.keras.layers.Concatenate()
    self.block1 = OptimizedBlock(ch, ksize=3)
    self.block2 = Block(ch, ch * 2, ksize=3, downsample=True)
    # self.sn_block = SNNonLocalBlock(ch * 2)
    self.block3 = Block(ch * 2, ch * 4, ksize=3, downsample=True)
    self.block4 = Block(ch * 4, ch * 8, ksize=3, downsample=True)
    self.block5 = Block(ch * 8, ch * 16, ksize=3, downsample=True)
    self.linear = SNDense(units=1, kernel_initializer=initializer)

    if n_classes > 0:
      self.embeddings = SNEmbeeding(embedding_size=ch * 16, n_classes=10, kernel_initializer=initializer)

  def call(self, x, y, *args, **kwargs):
    h = x
    h = self.block1(h, **kwargs)
    h = self.block2(h, **kwargs)
    # h = self.sn_block(h)
    h = self.block3(h, **kwargs)
    h = self.block4(h, **kwargs)
    h = self.block5(h, **kwargs)
    h = self.activation(h)
    h = tf.reduce_sum(h, axis=(1, 2))
    output = self.linear(h, **kwargs)
    if y is not None:
      embed = self.embeddings(y, **kwargs)
      output += tf.reduce_sum(h * embed, axis=1, keepdims=True)

    return h
