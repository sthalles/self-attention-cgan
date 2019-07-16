import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
import numpy as np
from discriminator.resnet_discriminator import ResnetDiscriminator
from generator.unet_generator_v2 import UNetGenerator

N_CLASSES=10
IMAGE_SIZE=128
BATCH_SIZE=2

L_channel = tf.random.uniform(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32)
AB_channel = tf.random.uniform(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 2), dtype=tf.float32)

fake_labels = np.random.randint(low=0, high=10, size=(BATCH_SIZE)).astype(np.int32)
assert fake_labels.shape == (BATCH_SIZE,), "Error with the expected labels shape"

lab_input = tf.concat((L_channel, AB_channel), axis=-1)
assert lab_input.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), "Error with the lab_image shape"

kwargs={'training': True}
generator = UNetGenerator(ch=16, out_channels=2)

gen_output = generator(L_channel, sn_update=True, **kwargs)
assert gen_output.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 2), "Error in the generator output shape"

discriminator = ResnetDiscriminator(ch=16, n_classes=N_CLASSES)

disc_fake = discriminator(lab_input, fake_labels, sn_update=True, **kwargs)
assert disc_fake.shape == (BATCH_SIZE, 1), "Error in the discriminator output shape"

disc_fake = discriminator(lab_input, fake_labels, sn_update=False, **kwargs)
assert disc_fake.shape == (BATCH_SIZE, 1), "Error in the discriminator output shape"
