import tensorflow as tf
from tensorflow.python.ops import variable_scope


def uNet(inputs, is_training):
    with variable_scope.variable_scope("Unet_1024", 'Unet_1024', [inputs]):
        down0b = tf.layers.conv2d(inputs, 8, 3, padding="same")
        down0b = tf.layers.batch_normalization(down0b, training=is_training)
        down0b = tf.nn.relu(down0b)
        down0b = tf.layers.conv2d(down0b, 8, 3, padding="same")
        down0b = tf.layers.batch_normalization(down0b, training=is_training)
        down0b = tf.nn.relu(down0b)
        down0b_pool = tf.layers.max_pooling2d(down0b, 2, 2, padding="valid")
        # 512

        down0a = tf.layers.conv2d(down0b_pool, 16, 3, padding="same")
        down0a = tf.layers.batch_normalization(down0a, training=is_training)
        down0a = tf.nn.relu(down0a)
        down0a = tf.layers.conv2d(down0a, 16, 3, padding="same")
        down0a = tf.layers.batch_normalization(down0a, training=is_training)
        down0a = tf.nn.relu(down0a)
        down0a_pool = tf.layers.max_pooling2d(down0a, 2, 2, padding="valid")
        # 256

        down0 = tf.layers.conv2d(down0a_pool, 32, 3, padding="same")
        down0 = tf.layers.batch_normalization(down0, training=is_training)
        down0 = tf.nn.relu(down0)
        down0 = tf.layers.conv2d(down0, 32, 3, padding="same")
        down0 = tf.layers.batch_normalization(down0, training=is_training)
        down0 = tf.nn.relu(down0)
        down0_pool = tf.layers.max_pooling2d(down0, 2, 2, padding="valid")
        # 128

        down1 = tf.layers.conv2d(down0_pool, 64, 3, padding="same")
        down1 = tf.layers.batch_normalization(down1, training=is_training)
        down1 = tf.nn.relu(down1)
        down1 = tf.layers.conv2d(down1, 64, 3, padding="same")
        down1 = tf.layers.batch_normalization(down1, training=is_training)
        down1 = tf.nn.relu(down1)
        down1_pool = tf.layers.max_pooling2d(down1, 2, 2, padding="valid")
        # 64

        down2 = tf.layers.conv2d(down1_pool, 128, 3, padding="same")
        down2 = tf.layers.batch_normalization(down2, training=is_training)
        down2 = tf.nn.relu(down2)
        down2 = tf.layers.conv2d(down2, 128, 3, padding="same")
        down2 = tf.layers.batch_normalization(down2, training=is_training)
        down2 = tf.nn.relu(down2)
        down2_pool = tf.layers.max_pooling2d(down2, 2, 2, padding="valid")
        # 32

        down3 = tf.layers.conv2d(down2_pool, 256, 3, padding="same")
        down3 = tf.layers.batch_normalization(down3, training=is_training)
        down3 = tf.nn.relu(down3)
        down3 = tf.layers.conv2d(down3, 256, 3, padding="same")
        down3 = tf.layers.batch_normalization(down3, training=is_training)
        down3 = tf.nn.relu(down3)
        down3_pool = tf.layers.max_pooling2d(down3, 2, 2, padding="valid")
        # 16

        down4 = tf.layers.conv2d(down3_pool, 512, 3, padding="same")
        down4 = tf.layers.batch_normalization(down4, training=is_training)
        down4 = tf.nn.relu(down4)
        down4 = tf.layers.conv2d(down4, 512, 3, padding="same")
        down4 = tf.layers.batch_normalization(down4, training=is_training)
        down4 = tf.nn.relu(down4)
        down4_pool = tf.layers.max_pooling2d(down4, 2, 2, padding="valid")
        # 8

        center = tf.layers.conv2d(down4_pool, 1024, 3, padding="same")
        center = tf.layers.batch_normalization(center, training=is_training)
        center = tf.nn.relu(center)
        center = tf.layers.conv2d(center, 1024, 3, padding="same")
        center = tf.layers.batch_normalization(center, training=is_training)
        center = tf.nn.relu(center)
        # center

        up4 = tf.image.resize_nearest_neighbor(center, [16, 16])
        up4 = tf.concat([up4, down4], axis=3)
        up4 = tf.layers.conv2d(up4, 512, 3, padding="same")
        up4 = tf.layers.batch_normalization(up4, training=is_training)
        up4 = tf.nn.relu(up4)
        up4 = tf.layers.conv2d(up4, 512, 3, padding="same")
        up4 = tf.layers.batch_normalization(up4, training=is_training)
        up4 = tf.nn.relu(up4)
        up4 = tf.layers.conv2d(up4, 512, 3, padding="same")
        up4 = tf.layers.batch_normalization(up4, training=is_training)
        up4 = tf.nn.relu(up4)
        # 16

        up3 = tf.image.resize_nearest_neighbor(up4, [32, 32])
        up3 = tf.concat([up3, down3], axis=3)
        up3 = tf.layers.conv2d(up3, 256, 3, padding="same")
        up3 = tf.layers.batch_normalization(up3, training=is_training)
        up3 = tf.nn.relu(up3)
        up3 = tf.layers.conv2d(up3, 256, 3, padding="same")
        up3 = tf.layers.batch_normalization(up3, training=is_training)
        up3 = tf.nn.relu(up3)
        up3 = tf.layers.conv2d(up3, 256, 3, padding="same")
        up3 = tf.layers.batch_normalization(up3, training=is_training)
        up3 = tf.nn.relu(up3)
        # 32

        up2 = tf.image.resize_nearest_neighbor(up3, [64, 64])
        up2 = tf.concat([up2, down2], axis=3)
        up2 = tf.layers.conv2d(up2, 128, 3, padding="same")
        up2 = tf.layers.batch_normalization(up2, training=is_training)
        up2 = tf.nn.relu(up2)
        up2 = tf.layers.conv2d(up2, 128, 3, padding="same")
        up2 = tf.layers.batch_normalization(up2, training=is_training)
        up2 = tf.nn.relu(up2)
        up2 = tf.layers.conv2d(up2, 128, 3, padding="same")
        up2 = tf.layers.batch_normalization(up2, training=is_training)
        up2 = tf.nn.relu(up2)
        # 64

        up1 = tf.image.resize_nearest_neighbor(up2, [128, 128])
        up1 = tf.concat([up1, down1], axis=3)
        up1 = tf.layers.conv2d(up1, 64, 3, padding="same")
        up1 = tf.layers.batch_normalization(up1, training=is_training)
        up1 = tf.nn.relu(up1)
        up1 = tf.layers.conv2d(up1, 64, 3, padding="same")
        up1 = tf.layers.batch_normalization(up1, training=is_training)
        up1 = tf.nn.relu(up1)
        up1 = tf.layers.conv2d(up1, 64, 3, padding="same")
        up1 = tf.layers.batch_normalization(up1, training=is_training)
        up1 = tf.nn.relu(up1)
        # 128

        up0 = tf.image.resize_nearest_neighbor(up1, [256, 256])
        up0 = tf.concat([up0, down0], axis=3)
        up0 = tf.layers.conv2d(up0, 32, 3, padding="same")
        up0 = tf.layers.batch_normalization(up0, training=is_training)
        up0 = tf.nn.relu(up0)
        up0 = tf.layers.conv2d(up0, 32, 3, padding="same")
        up0 = tf.layers.batch_normalization(up0, training=is_training)
        up0 = tf.nn.relu(up0)
        up0 = tf.layers.conv2d(up0, 32, 3, padding="same")
        up0 = tf.layers.batch_normalization(up0, training=is_training)
        up0 = tf.nn.relu(up0)
        # 256

        up0a = tf.image.resize_nearest_neighbor(up0, [512, 512])
        up0a = tf.concat([up0a, down0a], axis=3)
        up0a = tf.layers.conv2d(up0a, 16, 3, padding="same")
        up0a = tf.layers.batch_normalization(up0a, training=is_training)
        up0a = tf.nn.relu(up0a)
        up0a = tf.layers.conv2d(up0a, 16, 3, padding="same")
        up0a = tf.layers.batch_normalization(up0a, training=is_training)
        up0a = tf.nn.relu(up0a)
        up0a = tf.layers.conv2d(up0a, 16, 3, padding="same")
        up0a = tf.layers.batch_normalization(up0a, training=is_training)
        up0a = tf.nn.relu(up0a)
        # 512

        up0b = tf.image.resize_nearest_neighbor(up0a, [1024, 1024])
        up0b = tf.concat([up0b, down0b], axis=3)
        up0b = tf.layers.conv2d(up0b, 8, 3, padding="same")
        up0b = tf.layers.batch_normalization(up0b, training=is_training)
        up0b = tf.nn.relu(up0b)
        up0b = tf.layers.conv2d(up0b, 8, 3, padding="same")
        up0b = tf.layers.batch_normalization(up0b, training=is_training)
        up0b = tf.nn.relu(up0b)
        up0b = tf.layers.conv2d(up0b, 8, 3, padding="same")
        up0b = tf.layers.batch_normalization(up0b, training=is_training)
        up0b = tf.nn.relu(up0b)
        # 1024

        classify = tf.layers.conv2d(up0b, 1, 1)

    return classify
