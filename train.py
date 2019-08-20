# encoding:utf-8
import tensorflow as tf
import os
from config import *
from datautils import data_generator
# import data_origin

import numpy as np
import time
import net
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(argv):
    train_path = train_path_list[argv.train_tag]
    tag = argv.logs_tag
    batch_size = argv.batch_size
    patch_size = argv.patch_size

    learning_rate = argv.learning_rate
    lr_decay = argv.lr_decay
    decay_steps = argv.decay_step

    max_step = argv.max_step
    save_step = argv.save_step
    log_step = argv.log_step
    if_resort = argv.if_resort

    checkpoint_path = logs_list[tag]

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    y_ = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_labels')

    # y = net.model(x, True, regular=regularization_rate, use_detail=use_detail, use_se=use_se)
    y = net.choice(x, tag, True)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    mse_loss = tf.reduce_mean(tf.square(y - y_))
    total_loss = tf.add_n([mse_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    psnr = tf.reduce_mean(tf.image.psnr(y, y_, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(y, y_, max_val=1.0))

    tf.summary.scalar('model_loss', mse_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('psnr', psnr)
    tf.summary.scalar('ssim', ssim)

    lr_adam = tf.train.exponential_decay(learning_rate, global_step, decay_steps=decay_steps, decay_rate=lr_decay, staircase=True)
    lr_sgd = tf.Variable(tf.constant(0.0001), dtype=tf.float32)

    tf.summary.scalar('lr_adam', lr_adam)
    tf.summary.scalar('lr_adam', lr_sgd)
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step)
    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    with tf.control_dependencies([update_ops]):
        train_adam_op = tf.train.AdamOptimizer(lr_adam).minimize(total_loss, global_step)
        train_sgd_op = tf.train.GradientDescentOptimizer(lr_sgd).minimize(total_loss, global_step)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()
    # summary_train_op = tf.summary.merge([ml_sum, tl_sum, lr_sum])
    # summary_test_op = tf.summary.merge([ps_sum, ss_sum])

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU setting
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if tf.gfile.Exists(checkpoint_path) and if_resort:
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            current_step = int(checkpoint.split('-')[1]) + 1
            print('continue training from previous checkpoint', checkpoint_path)
            saver.restore(sess, checkpoint)
        else:
            if tf.gfile.Exists(checkpoint_path):
                tf.gfile.DeleteRecursively(checkpoint_path)
            print('new training for ', checkpoint_path)
            tf.gfile.MakeDirs(checkpoint_path)
            sess.run(init)
            current_step = 0

        summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

        data_gene = data_generator.get_batch(num_workers=8, data_path=train_path,
                                             input_size=patch_size, batch_size=batch_size)

        start = time.time()
        data_time = []

        for step in range(current_step, max_step+1):

            data_start = time.time()
            data = next(data_gene)
            data_time.append(time.time() - data_start)

            if step < 100000:
                _, mse_l, total_l, p, s, lr = sess.run([train_adam_op, mse_loss, total_loss,  psnr, ssim, lr_adam],
                                                       feed_dict={x: data[0], y_: data[1]})
            else:
                if step == 300000:
                    sess.run(tf.assign(lr_sgd, lr_sgd.eval() / 10))
                _, mse_l, total_l, p, s, lr = sess.run([train_sgd_op, mse_loss, total_loss, psnr, ssim, lr_sgd],
                                                       feed_dict={x: data[0], y_: data[1]})

            if np.isnan(total_l):
                print('loss diverged, stop training')
                break
            if step % log_step == 0:

                avg_time_per_step = (time.time() - start) / log_step
                avg_examples_per_second = (log_step * batch_size) / (time.time() - start)
                avg_time_data = sum(data_time) / log_step

                start = time.time()
                data_time = []
                print('{:06d}, ml {:.5f}, tl {:.5f}, {:.3f}s, {:.3f}d, {:.2f}e, p {:.4f}, s {:.4f}, lr {:.6f}'.format(
                    step, mse_l, total_l, avg_time_per_step, avg_time_data, avg_examples_per_second, p, s, lr))
                summary_str = sess.run(summary_op, feed_dict={x: data[0], y_: data[1]})
                summary_writer.add_summary(summary_str, global_step=step)

            if step % save_step == 0:
                saver.save(sess, checkpoint_path + 'model.ckpt', global_step=step)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
