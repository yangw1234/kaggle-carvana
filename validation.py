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


def calc_dice_coff(logits, masks):
    pred = tf.nn.sigmoid(logits)
    final_pred = resize_image(pred, [1280, 1918], DATA_FORMAT)
    if DATA_FORMAT == "NCHW":
        masks = tf.transpose(masks, perm=[0, 3, 2, 1])
    final_mask = tf.to_float(masks)
    inter = tf.reduce_sum(final_pred * final_mask)
    dice_coff = (2.0 * tf.to_float(inter) + 1.0) / (tf.to_float(tf.reduce_sum(final_pred)) + tf.to_float(tf.reduce_sum(final_mask)) + 1.0)
    return dice_coff


def bce_dice_loss(logits, masks):
    probability = tf.nn.sigmoid(logits)
    y = tf.cast(masks, tf.float32)
    # calc weights
    if DATA_FORMAT == "NCHW":
        a = tf.layers.average_pooling2d(y, 11, 1, padding="same", data_format="channels_first")
    else:
        a = tf.layers.average_pooling2d(y, 11, 1, padding="same", data_format="channels_last")
    ind = tf.cast(tf.cast(a > 0.01, tf.int32) * tf.cast(a < 0.99, tf.int32), tf.float32)
    weights = tf.ones(a.shape)
    w0 = tf.reduce_sum(weights)
    weights = weights + ind * 2.0
    w1 = tf.reduce_sum(weights)
    weights = weights / w1 * w0

    weights_2 = weights * weights

    intersection = tf.reduce_sum(weights_2 * y * probability)
    dice_coff_loss = 1 - (2. * intersection + 1.0) / (
        tf.reduce_sum(weights_2 * y) + tf.reduce_sum(weights_2 * probability) + 1.0)
    bce_loss = tf.reduce_mean(
        tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=y, weights=weights))
    all_loss = bce_loss + dice_coff_loss

    return all_loss

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, images, masks):
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
    mask_batch = resize_image_and_transpose(masks, size=[1024, 1024], data_format=DATA_FORMAT)
    from tensorflow.python.ops import init_ops
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device='/cpu:0'):
        with slim.arg_scope([slim.conv2d], weights_initializer=init_ops.glorot_uniform_initializer()):
            logits = uNet(image_batch, has_batch_norm=True, data_format=DATA_FORMAT)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    all_loss = bce_dice_loss(logits, mask_batch)

    dice_coff = calc_dice_coff(logits, masks)

    return all_loss, dice_coff


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def validation():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        ids = get_image_ids()

        validation_ids = ids[0:100]

        validation_size = len(validation_ids) * 16

        num_batches_per_epoch = (validation_size / FLAGS.batch_size)

        images, masks = get_inputs(validation_ids, "Validation_Data", False)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, masks], capacity=2 * FLAGS.num_gpus)

        losses = []
        dice_coffs = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("unet", i)) as scope:
                        image_batch, mask_batch = batch_queue.dequeue()

                        loss, dice_coff = tower_loss(scope, image_batch, mask_batch)

                        losses.append(loss)
                        dice_coffs.append(dice_coff)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

        avg_loss = tf.div(tf.add_n(losses), len(losses), name="avg_loss")
        avg_dice_coff = tf.div(tf.add_n(dice_coffs), len(dice_coffs), name="avg_dice_coff")

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

        init_step = global_step.eval(session=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        loss_acc = 0.0
        dice_coff_acc = 0.0
        for step in xrange(num_batches_per_epoch / FLAGS.num_gpus):
            start_time = time.time()
            loss_value, dice_coff_value = sess.run([avg_loss, avg_dice_coff])
            duration = time.time() - start_time

            loss_acc = loss_acc + loss_value
            dice_coff_acc = dice_coff_acc + dice_coff_value

            num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus

            format_str = ('%s: global_step %d step %d, loss = %.4f, dice_coff = %.4f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            logging.info(format_str % (
            datetime.now(), init_step, step, loss_value,
            dice_coff_value, examples_per_sec, sec_per_batch))

        val_avg_loss = loss_acc / (num_batches_per_epoch / FLAGS.num_gpus)
        val_avg_dice_coff = dice_coff_acc / (num_batches_per_epoch / FLAGS.num_gpus)
        sry = tf.Summary()
        sry.value.add(tag="validation_dice_coff", simple_value=val_avg_dice_coff)
        sry.value.add(tag="validation_loss", simple_value=val_avg_dice_coff)
        summary_writer.add_summary(sry, global_step=init_step)
        logging.info("validation, global_step: %s, loss: %s dice_coff: %s" % (init_step, val_avg_loss, val_avg_dice_coff))

        summary_writer.add_summary(sry, init_step)


def get_inputs(ids, scope, is_training):
    with tf.name_scope(scope):
        image, mask = get_image_and_label(ids)
        image, mask = propressing(image, mask, is_training)
        image_batch, mask_batch = tf.train.batch([image, mask],
                                                 batch_size=FLAGS.batch_size,
                                                 capacity=400, num_threads=8)
        return image_batch, mask_batch


def main(argv=None):  # pylint: disable=unused-argument
    validation()


if __name__ == '__main__':
    tf.app.run()
