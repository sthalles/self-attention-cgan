import tensorflow as tf
from layers.conv_sn import SNConv2D


class Block(tf.keras.Model):
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding='SAME',
                 activation=tf.keras.layers.ReLU(), upsample=False, n_classes=0):
        super(Block, self).__init__()
        initializer = tf.keras.initializers.glorot_uniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.unpooling_2d = tf.keras.layers.UpSampling2D()

        self.c1 = SNConv2D(hidden_channels, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)
        self.c2 = SNConv2D(out_channels, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)
        if n_classes > 0:
            pass
            # self.b1 = CategoricalConditionalBatchNormalization(in_channels, n_cat=n_classes)
            # self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, n_cat=n_classes)
        else:
            self.b1 = tf.keras.layers.BatchNormalization()
            self.b2 = tf.keras.layers.BatchNormalization()
        if self.learnable_sc:
            self.c_sc = SNConv2D(out_channels, kernel_size=1, padding="VALID", kernel_initializer=initializer)

    def residual(self, x, y=None, sn_update=None, **kwargs):
        assert sn_update is not None, "Specify the 'sn_update' parameter"
        h = x

        if y is not None:
            h = self.b1(h, y, **kwargs)
        else:
            h = self.b1(h, **kwargs)

        h = self.activation(h)

        if self.upsample:
            h = self.unpooling_2d(h)

        h = self.c1(h, sn_update=sn_update)

        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h, **kwargs)

        h = self.activation(h)
        h = self.c2(h, sn_update=sn_update)
        return h

    def shortcut(self, x, sn_update):
        if self.learnable_sc:

            if self.upsample:
                x = self.unpooling_2d(x)

            x = self.c_sc(x, sn_update=sn_update)
            return x
        else:
            return x

    def __call__(self, x, y=None, sn_update=None, **kwargs):
        return self.residual(x, y, sn_update=sn_update, **kwargs) + self.shortcut(x, sn_update=sn_update)
