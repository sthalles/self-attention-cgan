import tensorflow as tf
from layers.conv_sn import SNConv2D
from layers.transpose_conv_sn import SNTransposeConv2D


class DownSample(tf.keras.Model):
    def __init__(self, out_filters, kernel_size, activation, apply_batchnorm=True):
        super(DownSample, self).__init__()
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.keras.initializers.glorot_uniform()

        self.conv = SNConv2D(out_filters, kernel_size, strides=2, padding='SAME',
                             kernel_initializer=initializer, use_bias=not apply_batchnorm)

        if apply_batchnorm:
            self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, sn_update, **kwargs):
        net = self.conv(x, sn_update=sn_update)

        if self.apply_batchnorm:
            net = self.bn(net, **kwargs)

        net = self.activation(net)
        return net


class UpSample(tf.keras.Model):
    def __init__(self, in_filters, kernel_size, activation, apply_dropout=False):
        super(UpSample, self).__init__()
        initializer = tf.keras.initializers.glorot_uniform()
        self.activation = activation
        self.apply_dropout = apply_dropout
        self.conv = SNTransposeConv2D(in_filters, kernel_size, strides=2,
                                      padding='SAME',
                                      kernel_initializer=initializer,
                                      use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.SpatialDropout2D(0.5)

    def call(self, x, sn_update, output_shape, **kwargs):
        net = self.conv(x, sn_update=sn_update, output_shape=output_shape)
        net = self.bn(net, **kwargs)
        if self.apply_dropout:
            net = self.dropout(net, **kwargs)
        return self.activation(net)