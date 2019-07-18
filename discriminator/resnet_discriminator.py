import tensorflow as tf
from discriminator.resblocks import OptimizedBlock, Block
from layers.embedding_sn import SNEmbeeding
from layers.dense_sn import SNDense
from layers.sn_non_local_block import SNNonLocalBlock

class ResnetDiscriminator(tf.keras.Model):
    def __init__(self, ch, activation=tf.keras.layers.ReLU(), n_classes=0):
        super(ResnetDiscriminator, self).__init__()
        initializer = tf.keras.initializers.glorot_uniform()
        self.activation = activation
        self.concat = tf.keras.layers.Concatenate()
        self.block1 = OptimizedBlock(ch, ksize=3)
        self.block2 = Block(ch, ch * 2, ksize=3, downsample=True)
        self.sn_block = SNNonLocalBlock(ch * 2)
        self.block3 = Block(ch * 2, ch * 4, ksize=3, downsample=True)
        self.block4 = Block(ch * 4, ch * 8, ksize=3, downsample=True)
        self.block5 = Block(ch * 8, ch * 16, ksize=3, downsample=True)
        self.linear = SNDense(units=1, kernel_initializer=initializer)

        if n_classes > 0:
            self.embeddings = SNEmbeeding(embedding_size=ch * 16, n_classes=n_classes, kernel_initializer=initializer)

    def call(self, x, y, sn_update, **kwargs):
        h = x
        h = self.block1(h, sn_update=sn_update, **kwargs)
        h = self.block2(h, sn_update=sn_update, **kwargs)
        h = self.sn_block(h, sn_update=sn_update)
        h = self.block3(h, sn_update=sn_update, **kwargs)
        h = self.block4(h, sn_update=sn_update, **kwargs)
        h = self.block5(h, sn_update=sn_update, **kwargs)
        h = self.activation(h)
        h = tf.reduce_sum(h, axis=(1, 2))
        output = self.linear(h, sn_update=sn_update)
        if y is not None:
            embed = self.embeddings(y, sn_update=sn_update, )
            output += tf.reduce_sum(h * embed, axis=1, keepdims=True)

        return output
