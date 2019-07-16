import tensorflow as tf
# from source.sn_non_local_block import SNNonLocalBlock
from layers.conv_sn import SNConv2D
from layers.transpose_conv_sn import SNTransposeConv2D
from layers.netblocks import UpSample, DownSample


class UNetGenerator(tf.keras.Model):
  def __init__(self, ch, n_classes, activation=tf.keras.layers.ReLU()):
    super(UNetGenerator, self).__init__()
    # encoder
    self.down1 = DownSample(ch, 3, apply_batchnorm=False, activation=activation)
    self.down2 = DownSample(ch*2, 3, activation=activation)
    # self.enc_attention = SNNonLocalBlock(ch*2)
    self.down3 = DownSample(ch*2, 3, activation=activation)
    self.down4 = DownSample(ch*4, 3, activation=activation)
    self.down5 = DownSample(ch*4, 3, activation=activation)
    self.down6 = DownSample(ch*8, 3, activation=activation)
    self.down7 = DownSample(ch*8, 3, activation=activation)

    # decoder
    self.up1 = UpSample(ch*8, 3, apply_dropout=True, activation=activation)
    self.up2 = UpSample(ch*4, 3, apply_dropout=True, activation=activation)
    self.up3 = UpSample(ch*4, 3, apply_dropout=True, activation=activation)
    self.up4 = UpSample(ch*2, 3, activation=activation)
    self.up5 = UpSample(ch*2, 3, activation=activation)
    # self.dec_attention = SNNonLocalBlock(ch*2)
    self.up6 = UpSample(ch, 3, activation=activation)

    self.concat = tf.keras.layers.Concatenate()

    initializer = tf.keras.initializers.glorot_uniform()

    self.conv = SNTransposeConv2D(n_classes, kernel_size=4, strides=2,
                                  padding='SAME',
                                  kernel_initializer=initializer)

  def call(self, x, *args, **kwargs):
    # encoder forward
    down1 = self.down1(x, **kwargs)
    down2 = self.down2(down1, **kwargs)
    enc_attention = down2 #self.enc_attention(down2)
    down3 = self.down3(enc_attention, **kwargs)
    down4 = self.down4(down3, **kwargs)
    down5 = self.down5(down4, **kwargs)
    down6 = self.down6(down5, **kwargs)
    down7 = self.down7(down6, **kwargs)

    # decoder forward
    up1 = self.up1(down7, **kwargs)
    up1 = self.concat([up1, down6])

    up2 = self.up2(up1, **kwargs)
    up2 = self.concat([up2, down5])

    up3 = self.up3(up2, **kwargs)
    up3 = self.concat([up3, down4])

    up4 = self.up4(up3, **kwargs)
    up4 = self.concat([up4, down3])

    up5 = self.up5(up4, **kwargs)
    up5 = self.concat([up5, down2])

    dec_attention = up5 #self.dec_attention(up5)

    up6 = self.up6(dec_attention, **kwargs)
    up6 = self.concat([up6, down1])

    return tf.nn.tanh(self.conv(up6, **kwargs))
