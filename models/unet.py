import tensorflow as tf
from tensorflow.python.ops import variable_scope

slim = tf.contrib.slim

def down_layer(inputs, filters, kernel_size, has_batch_norm, has_pool, data_format):
    if data_format == "NHWC":
        channels_order = "channels_last"
    else:
        channels_order = "channels_first"

    down = slim.conv2d(inputs, filters, kernel_size, padding="same", data_format=data_format, activation_fn=None)
    if has_batch_norm:
        down = tf.contrib.layers.batch_norm(down, fused=True, data_format=data_format)
    down = tf.nn.relu(down)
    down = slim.conv2d(down, filters, kernel_size, padding="same", data_format=data_format, activation_fn=None)
    if has_batch_norm:
        down = tf.contrib.layers.batch_norm(down, fused=True, data_format=data_format)
    down = tf.nn.relu(down)
    if has_pool:
        down_pool = slim.max_pool2d(down, 2, 2, padding="valid", data_format=data_format)
    else:
        down_pool = None
    return down, down_pool

def up_layer(ups, downs, filters, kernel_size, output_size, has_batch_norm, data_format, up_sample_type=0):
    if data_format == "NHWC":
        channels_order = "channels_last"
        channel_dim = 3
    else:
        channels_order = "channels_first"
        channel_dim = 1

    if up_sample_type == 0:
        ups = resize_image(ups, size=output_size, data_format=data_format)
    else:
        ups = slim.conv2d_transpose(ups, filters, kernel_size, stride=2, data_format=data_format)

    ups = tf.concat([ups, downs], axis=channel_dim)

    ups = slim.conv2d(ups, filters, kernel_size, padding="same", data_format=data_format, activation_fn=None)
    if has_batch_norm:
        ups = tf.contrib.layers.batch_norm(ups, fused=True, data_format=data_format)
    ups = tf.nn.relu(ups)

    ups = slim.conv2d(ups, filters, kernel_size, padding="same", data_format=data_format, activation_fn=None)
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
    #image = tf.image.resize_bilinear(image, size=size)
    image = tf.image.resize_image_with_crop_or_pad(image, target_height=1280, target_width=1920)
    if data_format == "NCHW":
        image = tf.transpose(image, perm=[0, 3, 2, 1])
    return image


def uNet(inputs, has_batch_norm, data_format):

    with variable_scope.variable_scope("Unet", 'Unet', [inputs]):

        down0b, down0b_pool = down_layer(inputs, 8, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 512

        down0a, down0a_pool = down_layer(down0b_pool, 16, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 256

        down0, down0_pool = down_layer(down0a_pool, 32, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 128

        down1, down1_pool = down_layer(down0_pool, 64, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 64

        down2, down2_pool = down_layer(down1_pool, 128, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 32

        down3, down3_pool = down_layer(down2_pool, 256, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 16

        down4, down4_pool = down_layer(down3_pool, 512, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 8

        center, _ = down_layer(down4_pool, 1024, 3, has_batch_norm=has_batch_norm, has_pool=False, data_format=data_format)
        # center

        up4 = up_layer(center, down4, 512, 3, [20, 30], has_batch_norm=has_batch_norm, data_format=data_format)
        # 16

        up3 = up_layer(up4, down3, 256, 3, [40, 60], has_batch_norm=has_batch_norm, data_format=data_format)
        # 32

        up2 = up_layer(up3, down2, 128, 3, [80, 120], has_batch_norm=has_batch_norm, data_format=data_format)
        # 64

        up1 = up_layer(up2, down1, 64, 3, [160, 240], has_batch_norm=has_batch_norm, data_format=data_format)
        # 128

        up0 = up_layer(up1, down0, 32, 3, [320, 480], has_batch_norm=has_batch_norm, data_format=data_format)
        # 256

        up0a = up_layer(up0, down0a, 16, 3, [640, 960], has_batch_norm=has_batch_norm, data_format=data_format)
        # 512

        up0b = up_layer(up0a, down0b, 8, 3, [1280, 1920], has_batch_norm=has_batch_norm, data_format=data_format)
        # 1024

        classify = slim.conv2d(up0b, 1, 1, data_format=data_format, activation_fn=None)

    return classify

def uNet_v2(inputs, has_batch_norm, data_format):

    with variable_scope.variable_scope("Unet", 'Unet', [inputs]):

        down0b, down0b_pool = down_layer(inputs, 8, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 512

        down0a, down0a_pool = down_layer(down0b_pool, 16, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 256

        down0, down0_pool = down_layer(down0a_pool, 32, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 128

        down1, down1_pool = down_layer(down0_pool, 64, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 64

        down2, down2_pool = down_layer(down1_pool, 128, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 32

        down3, down3_pool = down_layer(down2_pool, 256, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 16

        down4, down4_pool = down_layer(down3_pool, 512, 3, has_batch_norm=has_batch_norm, has_pool=True, data_format=data_format)
        # 8

        center, _ = down_layer(down4_pool, 1024, 3, has_batch_norm=has_batch_norm, has_pool=False, data_format=data_format)
        # center

        up4 = up_layer(center, down4, 512, 3, [20, 30], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 16

        up3 = up_layer(up4, down3, 256, 3, [40, 60], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 32

        up2 = up_layer(up3, down2, 128, 3, [80, 120], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 64

        up1 = up_layer(up2, down1, 64, 3, [160, 240], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 128

        up0 = up_layer(up1, down0, 32, 3, [320, 480], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 256

        up0a = up_layer(up0, down0a, 16, 3, [640, 960], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 512

        up0b = up_layer(up0a, down0b, 8, 3, [1280, 1920], has_batch_norm=has_batch_norm, data_format=data_format, up_sample_type=1)
        # 1024
        if data_format == "NHWC":
            channels_order = "channels_last"
            channel_dim = 3
        else:
            channels_order = "channels_first"
            channel_dim = 1
        final_up = tf.concat([inputs, up0b], axis=channel_dim)

        classify = slim.conv2d(final_up, 4, 3, data_format=data_format, padding="SAME")
        classify = slim.conv2d(classify, 1, 1, data_format=data_format, activation_fn=None, padding="SAME")

    return classify
