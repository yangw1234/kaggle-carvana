import tensorflow as tf
from input import *
from model import *
from preprocessing import *

import matplotlib.pyplot as plt

batch_size = 10
min_after_dequeue = 10
capacity = 200

ids = get_image_ids()

training_ids = ids[100:]
validation_ids = ids[0:100]

training_size = len(training_ids) * 16
validation_size = len(validation_ids) * 16

n_batches = int(training_size / batch_size)

n_epoches = 20

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# get training data
with tf.name_scope("Training_Data"):
    training_image, training_mask = get_image_and_label(training_ids)
    training_image_batch, training_mask_batch = tf.train.shuffle_batch([training_image, training_mask],
                                                 batch_size=batch_size,
                                                 min_after_dequeue=min_after_dequeue,
                                                 capacity=capacity)

# get validation data
with tf.name_scope("Validation_Data"):
    validation_image, validation_mask = get_image_and_label(validation_ids)
    validation_image_batch, validation_mask_batch = tf.train.shuffle_batch([validation_image, validation_mask],
                                                 batch_size=batch_size,
                                                 min_after_dequeue=min_after_dequeue,
                                                 capacity=capacity)

is_training = tf.placeholder(tf.bool)

raw_image_batch = tf.cond(is_training, lambda: training_image_batch, lambda: validation_image_batch)
raw_mask_batch = tf.cond(is_training, lambda: training_mask_batch, lambda: validation_mask_batch)

image_batch, mask_batch = propressing(raw_image_batch, raw_mask_batch, (320, 480))

mask_batch = tf.squeeze(mask_batch, axis=3)

output = model(image_batch, is_training, scope="Model")

with tf.name_scope("Predictions"):
    softmax = tf.nn.softmax(output)
    pred = softmax[:,:,:,1:2]
    final_pred = tf.image.resize_bilinear(pred, [1280, 1918])
    final_mask = tf.to_float(tf.div(raw_mask_batch, 255))
    inter = tf.reduce_sum(final_pred * final_mask)
    val_dice_coff = (2.0 * tf.to_float(inter) + 1.0) / (tf.to_float(tf.reduce_sum(final_pred)) + tf.to_float(tf.reduce_sum(final_mask)) + 1.0)



with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=output, labels=mask_batch))
    all_loss = 0.5 * loss - val_dice_coff


starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.5, staircase=True)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(all_loss, global_step=global_step)

with tf.name_scope("Summary"):
    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('training_dice_coff', val_dice_coff)
    tf.summary.scalar('learning_rate', learning_rate)

    summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    writer = tf.summary.FileWriter("./graph", sess.graph)

    ckpt = tf.train.get_checkpoint_state("./checkpoints")

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    for index in range(initial_step, n_batches * n_epoches):
        # Retrieve a single instance:
        _, l, d, summary, p, m = sess.run([train_op, loss, val_dice_coff, summary_op, pred, mask_batch], feed_dict={is_training: True})
        print "step: %s, loss: %s, dice_coff: %s" % (index, l, d)

        if (index + 1) % 20 == 0:
            saver.save(sess, "./checkpoints/carvana", index)
            d_val, prediction, groud_truth = sess.run([val_dice_coff, final_pred, final_mask], feed_dict={is_training: False})
            print "validation, step: %s, dice_coff: %s" % (index, d_val)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(prediction[0, :, :, 0] > 0.5, cmap='gray')
            # ax[1].imshow(groud_truth[0, :, :, 0] > 0.5, cmap='gray')
            # plt.show()

        writer.add_summary(summary, global_step=index)

    coord.request_stop()
    coord.join(threads)