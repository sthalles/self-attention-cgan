import tensorflow as tf
import numpy as np
from discriminator.resnet_discriminator import ResnetDiscriminator
from generator.unet_generator_v2 import UNetGenerator

N_CLASSES=10
BATCH_SIZE=2

L_channel = tf.random.uniform(shape=(BATCH_SIZE, 128, 128, 1), dtype=tf.float32)
AB_channel = tf.random.uniform(shape=(BATCH_SIZE, 128, 128, 2), dtype=tf.float32)

fake_labels = np.random.randint(low=0, high=10, size=(BATCH_SIZE)).astype(np.int32)

lab_input = tf.concat((L_channel, AB_channel), axis=-1)


generator = UNetGenerator(ch=16,n_classes=10)
generator(L_channel, **{'training': True})


discriminator = ResnetDiscriminator(16, n_classes=N_CLASSES)
discriminator(lab_input, fake_labels, **{'training': True})