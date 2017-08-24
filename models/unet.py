import tensorflow as tf
from tensorflow.python.ops import variable_scope

slim = tf.contrib.slim

def down_layer(inputs, filters, kernel_size, has_batch_norm, has_pool, data_format):
    if data_format == "NHWC":
        channels_order = "channels_last"
    else:
        channels_order = "channels_first"

    down = slim.conv2d(inputs, filters, kernel_size, padding="same", data_format=data_format)
    if has_batch_norm:
        down = tf.contrib.layers.batch_norm(down, fused=True, data_format=data_format)
    down = tf.nn.relu(down)
    down = slim.conv2d(down, filters, kernel_size, padding="same", data_format=data_format)
    if has_batch_norm:
        down = tf.contrib.layers.batch_norm(down, fused=True, data_format=data_format)
    down = tf.nn.relu(down)
    if has_pool:
        down_pool = tf.layers.max_pooling2d(down, 2, 2, padding="valid", data_format=channels_order)
    else:
        down_pool = None
    return down, down_pool

def up_layer(ups, downs, filters, kernel_size, output_size, has_batch_norm, data_format):
    if data_format == "NHWC":
        channels_order = "channels_last"
        channel_dim = 3
    else:
        channels_order = "channels_first"
        channel_dim = 1

    ups = resize_image(ups, size=output_size, data_format=data_format)
    ups = tf.concat([ups, downs], axis=channel_dim)

    ups = tf.layers.conv2d(ups, filters, kernel_size, padding="same", data_format=channels_order)
    if has_batch_norm:
        ups = tf.contrib.layers.batch_norm(ups, fused=True, data_format=data_format)
    ups = tf.nn.relu(ups)

    ups = tf.layers.conv2d(ups, filters, kernel_size, padding="same", data_format=channels_order)
    if has_batch_norm:
        ups = tf.contrib.layers.batch_norm(ups, fused=True, data_format=data_format)
    ups = tf.nn.relu(ups)
    return ups

def resize_image(image, size, data_format):
    if data_format == "NCHW":
        image = tf.transpose(image, perm=[0, 3, 2, 1])
        image = tf.image.resize_bilinear(image, size=size)
        image = tf.transpose(image, perm=[0, 3, 2, 1])
    else:
        image = tf.image.resize_bilinear(image, size=size)
    return image

def resize_image_and_transpose(image, size, data_format):
    image = tf.image.resize_bilinear(image, size=size)
    if data_format == "NCHW":
        image = tf.transpose(image, perm=[0, 3, 2, 1])
    return image


def uNet(inputs, has_batch_norm, data_format):

    with variable_scope.variable_scope("Unet", 'Unet', [inputs]):

        # down0b, down0b_pool = down_layer(inputs, 8, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 512

        # down0a, down0a_pool = down_layer(down0b_pool, 16, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 256

        down0, down0_pool = down_layer(inputs, 8, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 128

        down1, down1_pool = down_layer(down0_pool, 16, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 64

        down2, down2_pool = down_layer(down1_pool, 32, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 32

        down3, down3_pool = down_layer(down2_pool, 64, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 16

        down4, down4_pool = down_layer(down3_pool, 128, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 8

        center, _ = down_layer(down4_pool, 256, 3, has_batch_norm=has_batch_norm, has_pool=False, data_format=data_format)
        # center

        up4 = up_layer(center, down4, 128, 3, [64, 64], has_batch_norm=has_batch_norm, data_format=data_format)
        # 16

        up3 = up_layer(up4, down3, 64, 3, [128, 128], has_batch_norm=has_batch_norm, data_format=data_format)
        # 32

        up2 = up_layer(up3, down2, 32, 3, [256, 256], has_batch_norm=has_batch_norm, data_format=data_format)
        # 64

        up1 = up_layer(up2, down1, 16, 3, [512, 512], has_batch_norm=has_batch_norm, data_format=data_format)
        # 128

        up0 = up_layer(up1, down0, 8, 3, [1024, 1024], has_batch_norm=has_batch_norm, data_format=data_format)
        # 256

        # up0a = up_layer(up0, down0a, 16, 3, [512, 512], has_batch_norm=has_batch_norm, data_format=data_format)
        # 512

        # up0b = up_layer(up0a, down0b, 8, 3, [1024, 1024], has_batch_norm=has_batch_norm, data_format=data_format)
        # 1024

        if data_format == "NCHW":
            classify = tf.layers.conv2d(up0, 1, 1, data_format="channels_first")
        else:
            classify = tf.layers.conv2d(up0, 1, 1, data_format="channels_last")

    return classify
