import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = "/home/yang/datasets/kaggle-carvana/"

def get_image_ids():
    file_name = DATA_DIR + "data/train_masks.csv"
    df_mask = pd.read_csv(file_name, usecols=['img'])
    ids_train = df_mask['img'].map(lambda s: s.split('_')[0]).unique()
    return ids_train

def get_test_image_files():
    file_name = DATA_DIR + "data/sample_submission.csv"
    df_mask = pd.read_csv(file_name, usecols=['img'])
    files = df_mask['img']
    return files

def get_train_image_files():
    file_name = DATA_DIR + "data/train_masks.csv"
    df_mask = pd.read_csv(file_name, usecols=['img'])
    files = df_mask['img'].map(lambda s: s.split('.')[0])
    return files


def read_images_from_disk(input_queue):
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    mask_content = tf.read_file(input_queue[1])
    mask = tf.image.decode_gif(mask_content)
    image = tf.image.resize_image_with_crop_or_pad(image, 1280, 1918)
    mask = tf.image.resize_image_with_crop_or_pad(mask, 1280, 1918)

    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.squeeze(mask, axis=0)

    image = tf.to_float(image)

    return image, mask

def get_image_and_label(ids_train):
    num = 16
    images = []
    masks = []
    for im in ids_train:
        for idx in range(1, num + 1):
            images.append(DATA_DIR + 'data/train/{}_{:02d}.jpg'.format(im, idx))
            masks.append(DATA_DIR + 'data/train_masks/{}_{:02d}_mask.gif'.format(im, idx))

    image_tensor = tf.convert_to_tensor(images)
    mask_tensor = tf.convert_to_tensor(masks)

    input_queue = tf.train.slice_input_producer([image_tensor, mask_tensor])

    image, mask = read_images_from_disk(input_queue)

    return image, mask

def get_test_image(files):
    images = []
    for im in files:
        images.append(DATA_DIR + 'data/test/' + im)

    image_tensor = tf.convert_to_tensor(images)
    input_queue = tf.train.slice_input_producer([image_tensor], shuffle=False)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 1280, 1918)
    image = tf.to_float(image)

    return image, input_queue[0]


