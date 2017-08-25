import tensorflow as tf
from models.unet import *
def propressing(image, mask, is_training):
    image = (image - 128.0) / 128.0
    mask = mask / 255
    # if is_training:
        # image = tf.image.random_flip_left_right(image)
        # mask = tf.image.random_flip_left_right(mask)

        # image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    mask = tf.to_int32(mask)
    return image, mask

def propressing_for_test(image, size, data_format):
    image = resize_image(image, size=size, data_format=data_format)

    image = (image - 128.0) / 128.0

    return image
