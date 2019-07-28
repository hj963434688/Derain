# encoding:utf-8
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import data_process
import numpy as np
import time
import net


batch_size = 64
patch_size = 49
number_readers = 8

lr = 0.01
lr_decay = 0.9
decay_steps = 10000
max_step = 800000
moving_decay = 0.997
regularization_rate = 1e-10

pre_train = False
pre_train_path = None

save_step = 2000
log_step = 50
val_step = 100
# save_checkpoint_path = 100

train_paths = ['./model', './expe_model/v_19420/', './expe_model/v_19422/', './expe_model/v_19424/',
               './expe_model/v_19501/', './expe_model/v_19505/', './expe_model/v_19508/',
               './expe_model/v_19510/', './expe_model/v_19515/', './expe_model/v_19527_my/',
               './expe_model/v_roi/']
# checkpoint_path = './expe_model/version_19420/'
checkpoint_path = train_paths[10]
model_name = 'model.ckpt'
use_se = False
use_detail = False
whether_resort = False


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not tf.gfile.Exists(checkpoint_path):
        tf.gfile.MakeDirs(checkpoint_path)
    else:
        if not whether_resort:
            tf.gfile.DeleteRecursively(checkpoint_path)
            tf.gfile.MakeDirs(checkpoint_path)

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    y_ = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_labels')

    # y = net.model(x, True, regular=regularization_rate, use_detail=use_detail, use_se=use_se)
    y = net.model_roi(x, True, regular=None)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # variable_averages = tf.train.ExponentialMovingAverage(moving_decay, global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())

    mse_loss = tf.reduce_mean(tf.square(y - y_))
    total_loss = tf.add_n([mse_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    psnr = tf.reduce_mean(tf.image.psnr(y, y_, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(y, y_, max_val=1.0))

    ml_sum = tf.summary.scalar('model_loss', mse_loss)
    tl_sum = tf.summary.scalar('total_loss', total_loss)
    ps_sum = tf.summary.scalar('psnr', psnr)
    ss_sum = tf.summary.scalar('ssim', ssim)

    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=decay_steps, decay_rate=lr_decay, staircase=True)
    lr_sum = tf.summary.scalar('learning_rate', learning_rate)

    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step)
    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    with tf.control_dependencies([train_step, update_ops]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver(tf.global_variables())

    summary_train_op = tf.summary.merge([ml_sum, tl_sum, lr_sum])
    summary_test_op = tf.summary.merge([ps_sum, ss_sum])

    summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if pre_train_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(pre_train_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    # v = tf.trainable_variables()
    # for i in v:
    #     print(i)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU setting
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if whether_resort:
            print('continue training from previous checkpoint', checkpoint_path)
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
            try:
                current_step = int(ckpt.split('-')[1]) + 1
            except:
                current_step = 0
            print(current_step)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            current_step = 0
            if pre_train_path is not None:
                variable_restore_op(sess)

        data_generator = data_process.get_batch(num_workers=number_readers, input_size=patch_size, batch_size=batch_size)

        start = time.time()
        gene_time = []
        train_time = []

        for step in range(current_step, max_step+1):
            # print(step, global_step.eval())

            gene_start = time.time()
            data = next(data_generator)

            gene_time.append(time.time() - gene_start)

            train_start = time.time()
            _, mse_l, total_l, p, s = sess.run([train_op, mse_loss, total_loss,  psnr, ssim],
                                               feed_dict={x: data[0], y_: data[1]})
            train_time.append(time.time() - train_start)

            if np.isnan(total_l):
                print('loss diverged, stop training')
                break
            if step % log_step == 0:
                # print(y.eval())
                avg_time_per_step = (time.time() - start) / log_step
                avg_examples_per_second = (log_step * batch_size) / (time.time() - start)
                start = time.time()
                avg_time_genebath = sum(gene_time) / log_step
                gene_time = []
                avg_time_train = sum(train_time) / log_step
                train_time = []
                print('Step {:06d}, model loss {:.5f}, total loss {:.5f}, {:.2f} seconds/step, {:.2f} examples/second, psnr{:.4f}, ssim{:.4f}'.format(
                        step, mse_l, total_l, avg_time_per_step, avg_examples_per_second, p, s))
                summary_str = sess.run(summary_train_op, feed_dict={x: data[0], y_: data[1]})
                summary_writer.add_summary(summary_str, global_step=step)

            if step % save_step == 0:
                saver.save(sess, checkpoint_path + 'model.ckpt', global_step=step)
                test_input, test_label = data_process.get_test_batch()
                # _, tl, summary_str = sess.run([total_loss, summary_op], feed_dict={x: data[0], y_: data[1]})
                summary_str = sess.run(summary_test_op, feed_dict={x: test_input, y_: test_label})
                summary_writer.add_summary(summary_str, global_step=step)
                # print(step, global_step.eval())
            # if step % val_step == 0:
            #     sess.run([mse_loss, total_loss, val_p, val_s])


if __name__ == '__main__':
    main()
