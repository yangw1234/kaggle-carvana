import tensorflow as tf
from input import *
from model import *
from preprocessing import *
from models.unet import *
import re
import logging
from datetime import datetime
import os
import time
from rle import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7,8"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('batch_size', 3,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='logs/validation.log',
                    filemode='w')

DATA_FORMAT = "NCHW"


def prediction(scope, images):
    """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

    # Build inference Graph.
    image_batch = resize_image_and_transpose(images, size=[1024, 1024], data_format=DATA_FORMAT)
    from tensorflow.python.ops import init_ops
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device='/cpu:0'):
        with slim.arg_scope([slim.conv2d], weights_initializer=init_ops.glorot_uniform_initializer()):
            logits = uNet(image_batch, has_batch_norm=True, data_format=DATA_FORMAT)
            pred = tf.nn.sigmoid(logits)
            final_pred = resize_image(pred, [1280, 1918], DATA_FORMAT)

    return final_pred


def inference():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        ids = get_image_ids()

        validation_ids = ids[0:100]

        validation_size = len(validation_ids) * 16

        num_batches_per_epoch = (validation_size / FLAGS.batch_size)

        images, file_names = get_test_inputs("Validation_Data", False)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, file_names], capacity=2 * FLAGS.num_gpus)

        outputQueue = tf.FIFOQueue(100, tf.float32)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("unet", i)) as scope:
                        image_batch, file_names = batch_queue.dequeue()

                        pred_mask = prediction(scope, image_batch)

                        outputQueue.enqueue([pred_mask, image_batch])

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()


        pred, file_name = outputQueue.dequeue()
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        submission_file = open("submission.csv", "w")
        submission_file.writelines("img,rle_mask\n")
        for step in xrange(num_batches_per_epoch):
            # Retrieve a single instance:
            start_time = time.time()
            pred_value, file_value = sess.run([pred, file_name])
            for i in range(0, FLAGS.batch_size):
                line = rle(pred_value[i, 0, :, :], file_value[i].split("/")[-1])
                submission_file.writelines(line + "\n")
            logging.info("num: %s batch, total %s batches, using %s seconds" % (
            step, num_batches_per_epoch, (time.time() - start_time)))

        submission_file.close()


def get_test_inputs(scope, is_training):
    with tf.name_scope(scope):
        files = get_test_image_files()
        test_image, file_name = get_test_image(files)
        test_image = propressing_for_test(test_image, [1024, 1024], is_training)
        image_batch, file_name = tf.train.batch([test_image, file_name],
                                                       batch_size=FLAGS.batch_size,
                                                       capacity=400, num_threads=8)

    return image_batch, file_name


def main(argv=None):  # pylint: disable=unused-argument
    inference()


if __name__ == '__main__':
    tf.app.run()
