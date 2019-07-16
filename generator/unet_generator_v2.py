import tensorflow as tf
# from source.sn_non_local_block import SNNonLocalBlock
from layers.conv_sn import SNConv2D
from layers.transpose_conv_sn import SNTransposeConv2D
from layers.netblocks import UpSample, DownSample


class UNetGenerator(tf.keras.Model):
    def __init__(self, ch, out_channels, activation=tf.keras.layers.ReLU()):
        super(UNetGenerator, self).__init__()
        # encoder
        self.down1 = DownSample(ch, 3, apply_batchnorm=False, activation=activation)
        self.down2 = DownSample(ch * 2, 3, activation=activation)
        # self.enc_attention = SNNonLocalBlock(ch*2)
        self.down3 = DownSample(ch * 2, 3, activation=activation)
        self.down4 = DownSample(ch * 4, 3, activation=activation)
        self.down5 = DownSample(ch * 4, 3, activation=activation)
        self.down6 = DownSample(ch * 8, 3, activation=activation)
        self.down7 = DownSample(ch * 8, 3, activation=activation)

        # decoder
        self.up1 = UpSample(ch * 8, 3, apply_dropout=True, activation=activation)
        self.up2 = UpSample(ch * 4, 3, apply_dropout=True, activation=activation)
        self.up3 = UpSample(ch * 4, 3, apply_dropout=True, activation=activation)
        self.up4 = UpSample(ch * 2, 3, activation=activation)
        self.up5 = UpSample(ch * 2, 3, activation=activation)
        # self.dec_attention = SNNonLocalBlock(ch*2)
        self.up6 = UpSample(ch, 3, activation=activation)

        self.concat = tf.keras.layers.Concatenate()

        initializer = tf.keras.initializers.glorot_uniform()

        self.conv = SNTransposeConv2D(out_channels, kernel_size=4, strides=2,
                                      padding='SAME',
                                      kernel_initializer=initializer)

    def call(self, x, sn_update, **kwargs):
        # encoder forward
        down1 = self.down1(x, sn_update=sn_update, **kwargs)
        down2 = self.down2(down1, sn_update=sn_update, **kwargs)
        enc_attention = down2  # self.enc_attention(down2)
        down3 = self.down3(enc_attention, sn_update=sn_update, **kwargs)
        down4 = self.down4(down3, sn_update=sn_update, **kwargs)
        down5 = self.down5(down4, sn_update=sn_update, **kwargs)
        down6 = self.down6(down5, sn_update=sn_update, **kwargs)
        down7 = self.down7(down6, sn_update=sn_update, **kwargs)

        # decoder forward
        h = self.up1(down7, sn_update=sn_update, output_shape=down6.shape, **kwargs)
        h = self.concat([h, down6])

        h = self.up2(h, sn_update=sn_update, output_shape=down5.shape, **kwargs)
        h = self.concat([h, down5])

        h = self.up3(h, sn_update=sn_update, output_shape=down4.shape, **kwargs)
        h = self.concat([h, down4])

        h = self.up4(h, sn_update=sn_update, output_shape=down3.shape, **kwargs)
        h = self.concat([h, down3])

        h = self.up5(h, sn_update=sn_update, output_shape=down2.shape, **kwargs)
        h = self.concat([h, down2])

        dec_attention = h  # self.dec_attention(up5)

        h = self.up6(dec_attention, sn_update=sn_update, output_shape=down1.shape, **kwargs)
        h = self.concat([h, down1])

        return tf.nn.tanh(self.conv(h, sn_update=sn_update))
