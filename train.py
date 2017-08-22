import tensorflow as tf
from input import *
from model import *
from preprocessing import *
from models.unet import *
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='train.log',
                filemode='w')

DATA_FORMAT = "NCHW"
batch_size = 1
min_after_dequeue = 10
capacity = 200

ids = get_image_ids()

training_ids = ids[100:]
validation_ids = ids[0:100]

training_size = len(training_ids) * 16
validation_size = len(validation_ids) * 16

n_batches = int(training_size / batch_size)

val_batches = int(validation_size / batch_size)

n_epoches = 40

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.device("cpu:0"):
    # get training data
    with tf.name_scope("Training_Data"):
        training_image, training_mask = get_image_and_label(training_ids)
        training_image, training_mask = propressing(training_image, training_mask, True)
        training_image_batch, training_mask_batch = tf.train.shuffle_batch([training_image, training_mask],
                                                     batch_size=batch_size,
                                                     min_after_dequeue=min_after_dequeue,
                                                     capacity=capacity, num_threads=4)

    # get validation data
    with tf.name_scope("Validation_Data"):
        validation_image, validation_mask = get_image_and_label(validation_ids)
        validation_image, validation_mask = propressing(validation_image, validation_mask, False)
        validation_image_batch, validation_mask_batch = tf.train.batch([validation_image, validation_mask],
                                                     batch_size=batch_size,
                                                     capacity=capacity)

    is_training = tf.placeholder(tf.bool)

    raw_image_batch = tf.cond(is_training, lambda: training_image_batch, lambda: validation_image_batch)
    raw_mask_batch = tf.cond(is_training, lambda: training_mask_batch, lambda: validation_mask_batch)

    image_batch = resize_image_and_transpose(raw_image_batch, size=[1024, 1024], data_format=DATA_FORMAT)
    mask_batch = resize_image_and_transpose(raw_mask_batch, size=[1024, 1024], data_format=DATA_FORMAT)

output = uNet(image_batch, has_batch_norm=True, data_format=DATA_FORMAT)

with tf.name_scope("Predictions"):
        pred = tf.nn.sigmoid(output)
        final_pred = resize_image(pred, [1280, 1918], DATA_FORMAT)
        if DATA_FORMAT == "NCHW":
            raw_mask_batch = tf.transpose(raw_mask_batch, perm=[0, 3, 2, 1])
        final_mask = tf.to_float(raw_mask_batch)
        inter = tf.reduce_sum(final_pred * final_mask)
        dice_coff = (2.0 * tf.to_float(inter) + 1.0) / (tf.to_float(tf.reduce_sum(final_pred)) + tf.to_float(tf.reduce_sum(final_mask)) + 1.0)


with tf.name_scope("Loss"):
        loss_pred = tf.nn.sigmoid(output)
        true_y = tf.cast(mask_batch, tf.float32)
        # calc weights
        if DATA_FORMAT == "NCHW":
            a = tf.layers.average_pooling2d(true_y, 11, 1, padding="same", data_format="channels_first")
        else:
            a = tf.layers.average_pooling2d(true_y, 11, 1, padding="same", data_format="channels_last")
        ind = tf.cast(tf.cast(a > 0.01, tf.int32) * tf.cast(a < 0.99, tf.int32), tf.float32)
        weights = tf.ones(a.shape)
        w0 = tf.reduce_sum(weights)
        weights = weights + ind*2.0
        w1 = tf.reduce_sum(weights)
        weights = weights/w1*w0

        weights_2 = weights * weights
        
        intersection = tf.reduce_sum(weights_2 * true_y * loss_pred)
        dice_coff_loss = 1 - (2. * intersection + 1.0) / (tf.reduce_sum(weights_2 * true_y) + tf.reduce_sum(weights_2 * loss_pred) + 1.0)
        bce_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=true_y, weights=weights))
        all_loss = bce_loss + dice_coff_loss

starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.8)
manually_learning_rate = 0.0001
train_op = tf.train.AdamOptimizer(manually_learning_rate).minimize(all_loss, global_step=global_step)

with tf.name_scope("Summary"):
    tf.summary.scalar('training_loss', all_loss)
    tf.summary.scalar('training_dice_coff', dice_coff)
    tf.summary.scalar('learning_rate', learning_rate)

    summary_op = tf.summary.merge_all()


init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(init)

    saver = tf.train.Saver()

    writer = tf.summary.FileWriter("./graph", sess.graph)

    ckpt = tf.train.get_checkpoint_state("./checkpoints")

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    for index in range(initial_step, n_batches * n_epoches):
        # Retrieve a single instance:
        import time
        start_time = time.time()
        _, l, d, summary = sess.run([train_op, all_loss, dice_coff, summary_op], feed_dict={is_training: True})
        logging.info("step: %s, loss: %s, dice_coff: %s, speed: %s img/s, total %s batches" % (index, l, d, 1.0 * batch_size / (time.time() - start_time), n_batches))

        if (index + 1) % n_batches == 0:
           saver.save(sess, "./checkpoints/carvana", index)

        if (index + 1) % n_batches == 0:
            acc = 0.0
            for val_index in range(0, val_batches):
                d_val = sess.run(dice_coff, feed_dict={is_training: False})
                logging.info("validation, step %s, dice_coff: %s" % (val_index, d_val))
                acc = acc + d_val
            acc = acc / val_batches
            sry = tf.Summary()
            sry.value.add(tag="validation_2_dice_coff", simple_value=acc)
            writer.add_summary(sry, global_step=(index+1)/n_batches)
            logging.info("validation, epoch: %s, dice_coff: %s" % ((index + 1)/n_batches, acc))
        writer.add_summary(summary, global_step=index)

    coord.request_stop()
    coord.join(threads)
