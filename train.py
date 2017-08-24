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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3,4"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_epoches', 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',
                    filemode='w')

DATA_FORMAT = "NCHW"


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

    tf.add_to_collection('losses', all_loss)

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
    
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device='/cpu:0'):
        logits = uNet(image_batch, has_batch_norm=True, data_format=DATA_FORMAT)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = bce_dice_loss(logits, mask_batch)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % "unet", '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss

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


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        ids = get_image_ids()

        training_ids = ids[100:]
        validation_ids = ids[0:100]

        training_size = len(training_ids) * 16
        validation_size = len(validation_ids) * 16

        num_batches_per_epoch = (training_size / FLAGS.batch_size)

        starter_learning_rate = 0.01
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.8)
        manually_learning_rate = 0.0001
        opt = tf.train.AdamOptimizer(manually_learning_rate)

        images, masks = get_training_inputs(training_ids)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, masks], capacity=2 * FLAGS.num_gpus)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("unet", i)) as scope:
                        image_batch, mask_batch = batch_queue.dequeue()

                        loss = tower_loss(scope, image_batch, mask_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        # tower_grads.append(grads)
                        
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

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

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_epoches * num_batches_per_epoch / FLAGS.num_gpus):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: epoch %d step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), (step + 1) / num_batches_per_epoch, (step + 1) % num_batches_per_epoch, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save the model checkpoint periodically.
            if (step + 1) % (num_batches_per_epoch/FLAGS.num_gpus) == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def get_training_inputs(training_ids):
    with tf.name_scope("Training_Data"):
        training_image, training_mask = get_image_and_label(training_ids)
        training_image, training_mask = propressing(training_image, training_mask, True)
        training_image_batch, training_mask_batch = tf.train.shuffle_batch([training_image, training_mask],
                                                                           batch_size=FLAGS.batch_size,
                                                                           min_after_dequeue=200,
                                                                           capacity=400, num_threads=16)
        return training_image_batch, training_mask_batch


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
