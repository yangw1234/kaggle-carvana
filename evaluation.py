import tensorflow as tf
from input import *
from model import *
from preprocessing import *
from rle import *
import logging
import time

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='eval.log',
                filemode='w')

batch_size = 16
min_after_dequeue = 10
capacity = 200

files = get_test_image_files()

test_size = len(files)

n_batches = int(test_size / batch_size)

# get training data
with tf.name_scope("Testing_Data"):
    test_image, file_name = get_test_image(files)
    [test_image_batch, file_name] = tf.train.batch([test_image, file_name],
                                                 batch_size=batch_size)

is_training = False

image_batch = propressing_for_test(test_image_batch, (160, 240))

output = model(image_batch, is_training, scope="Model")

with tf.name_scope("Predictions"):
    softmax = tf.nn.softmax(output)
    pred = softmax[:,:,:,1:2]
    final_pred = tf.image.resize_bilinear(pred, [1280, 1918])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state("./checkpoints_160_240")

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    submission_file = open("submission.csv", "w")
    submission_file.writelines("img,rle_mask\n")

    for index in range(0, n_batches):
        # Retrieve a single instance:
        start_time = time.time()
        pred, file = sess.run([final_pred, file_name])
        for i in range(0, batch_size):
            line = rle(pred[i, :, :, 0], file[i].split("/")[-1])
            submission_file.writelines(line + "\n")
        logging.info("num: %s batch, total %s batches, using %s seconds" % (index, n_batches, (time.time() - start_time)))

    submission_file.close()
    coord.request_stop()
    coord.join(threads)