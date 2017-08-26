import tensorflow as tf
from models.unet import *

def propressing(image, mask, is_training, is_aug=False):
    image = (image - 128.0) / 128.0
    mask = mask / 255
    if is_training and is_aug:
        image, mask = data_augmentation(image, mask)

    mask = tf.to_int32(mask)
    return image, mask

def propressing_for_test(image):
    # image = resize_image(image, size=size, data_format=data_format)

    image = (image - 128.0) / 128.0

    return image

def data_augmentation(image, mask):
    uniform_random = tf.random_ops.random_uniform([], 0, 1.0)
    mirror_cond = tf.math_ops.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(mask), lambda: mask)

    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image, mask

# from PIL import Image
# import numpy as np
# import time
#
# image = np.array(Image.open('/home/yang/datasets/kaggle-carvana/data/train/a5fea424990e_13.jpg'), dtype=np.uint8)
# mask = np.array(Image.open('/home/yang/datasets/kaggle-carvana/data/train/a5fea424990e_13.jpg'), dtype=np.uint8)
#
# image = tf.to_float(image)
#
# image = (image - 128.0) / 128.0
#
# image = data_augmentation(image, None)
#
# max = tf.reduce_max(image)
#
# min = tf.reduce_min(image)
#
# image = tf.cast((image - min) / (max - min) * 255, tf.uint8)
#
# with tf.Session() as sess:
#     for i in range(5):
#         image_value = sess.run(image)
#         Image.fromarray(image_value).show()
#         time.sleep(1)