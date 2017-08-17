import tensorflow as tf
def propressing(image, mask, size=(1280, 1920)):
    image = tf.image.resize_bilinear(image, size=size)
    mask = tf.image.resize_bilinear(mask, size=size)

    image = (image - 128.0) / 128.0

    mask = tf.to_int32(mask / 255)
    return image, mask

def propressing_for_test(image, size=(1280, 1920)):
    image = tf.image.resize_bilinear(image, size=size)

    image = (image - 128.0) / 128.0

    return image