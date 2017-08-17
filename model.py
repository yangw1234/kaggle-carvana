import tensorflow as tf
from models.inception_v1 import *




def model(image, training, scope):
    net, endpoints = inception_v1_base(image, final_endpoint="Mixed_4f")

    branch1 = tf.layers.conv2d_transpose(net, 512, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 20 * 30

    branch1 = tf.layers.conv2d_transpose(branch1, 256, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 40 * 60

    branch1 = tf.layers.conv2d_transpose(branch1, 128, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 80 * 120

    branch1 = tf.layers.conv2d_transpose(branch1, 64, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 160 * 240

    branch2 = tf.layers.conv2d_transpose(endpoints['Mixed_3c'], 256, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 40 * 60

    branch2 = tf.layers.conv2d_transpose(branch2, 128, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 80 * 120

    branch2 = tf.layers.conv2d_transpose(branch2, 64, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 160 * 240

    branch3 = tf.layers.conv2d_transpose(endpoints['Conv2d_2c_3x3'], 128, 3, strides=(2, 2), padding="same", activation=tf.nn.relu) # 80 * 120

    branch3 = tf.layers.conv2d_transpose(branch3, 64, 3, strides=(2, 2), padding="same", activation=tf.nn.relu)

    output = tf.concat([branch1, branch2, branch3], axis=3)

    output = tf.layers.conv2d(output, 2, 1, strides=(1, 1), padding="same")

    return output