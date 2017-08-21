import tensorflow as tf
def propressing(image, mask, size=(1280, 1920)):

    image = (image - 128.0) / 128.0
    mask = mask / 255

    # image = tf.image.random_flip_left_right(image)
    # mask = tf.image.random_flip_left_right(mask)

    # image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_hue(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    
    mask = tf.to_int32(mask)
    return image, mask

def propressing_for_test(image, size=(1280, 1920)):
    image = tf.image.resize_bilinear(image, size=size)

    image = (image - 128.0) / 128.0

    return image
