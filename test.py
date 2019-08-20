import tensorflow as tf
import sys
import time
import scipy.misc as sm
import net
import train as tr
from skimage.measure import compare_ssim, compare_psnr
import compu_psnr as cp
from config import *
from datautils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(argv):
    test_tag = argv.test_tag
    test_path = test_path_list[test_tag]
    tag = argv.logs_tag
    img_name = argv.img_name
    save_img = argv.save_img
    save_data = argv.save_data
    result_path = argv.result_path

    checkpoint_path = logs_list[tag]
    try:
        os.makedirs(result_path)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        y_ = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='label_images')

        y = net.choice(x, tag, False)
        mse = tf.reduce_mean(tf.square(y - y_))
        psnr = tf.image.psnr(y, y_, max_val=1.0)
        ssim = tf.image.ssim(y, y_, max_val=1.0)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('Restore from {}'.format(checkpoint))

                psnr_list = []
                ssim_list = []
                start = time.time()
                files = get_data_files(test_path)
                for i in range(len(files[0])):

                    image = sm.imread(files[0][i], mode='RGB')
                    image_net = image / 255
                    label = sm.imread(files[1][i], mode='RGB')
                    label_net = label / 255
                    # print(im.shape)
                    out, p, s, m, = sess.run([y, psnr, ssim, mse,], feed_dict={x: [image_net], y_: [label_net]})
                    out_float = out[0] * 255
                    out_int = out_float.astype(np.uint8)


                    # p1 = computer_psn(label, out_int, m=255.0,)
                    p1 = psnr2(out[0], label_net,)
                    # print(label_net.dtype, out_net.dtype)
                    s1 = compare_ssim(label_net.astype(np.float32), out[0], multichannel=True)

                    print(files[0][i], p, s, p1, s1)
                    psnr_list.append(p1)
                    ssim_list.append(s1)

                print(psnr_list)
                print(ssim_list)
                print('psnr:', sum(psnr_list) / len(psnr_list))
                print('ssim:', sum(ssim_list) / len(ssim_list))
                print('time:', time.time() - start)
                print('end')


def test_ing():

    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        y_ = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='label_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        y = net.model(x, False)

        mse = tf.reduce_mean(tf.square(y - y_))
        psnr = tf.reduce_mean(tf.image.psnr(y, y_, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(y, y_, max_val=1.0))

        # variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt = tf.train.latest_checkpoint(tr.checkpoint_path)
            # ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            # if ckpt_state is None:
            #     model_path = checkpoint_path
            # else:
            #     model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            if ckpt:
                saver.restore(sess, ckpt)
                print('Restore from {}'.format(ckpt))

                start = time.time()

                for i in range(100):
                    im, label = data_process.get_test_batch()

                    out, p, s, m = sess.run([y, psnr, ssim, mse], feed_dict={x: im, y_: label})

                    print(p, s, m)

                print(time.time() - start)
                print('end')


def test_one(name):

    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != 17:
            raise
    with tf.get_default_graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        y_ = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='label_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        y = net.model(x, False)

        mse = tf.reduce_mean(tf.square(y - y_))
        psnr = tf.image.psnr(y, y_, max_val=1.0)
        ssim = tf.image.ssim(y, y_, max_val=1.0)

        # variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt = tf.train.latest_checkpoint(tr.checkpoint_path)
            # ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            # if ckpt_state is None:
            #     model_path = checkpoint_path
            # else:
            #     model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            if ckpt:
                saver.restore(sess, ckpt)
                print('Restore from {}'.format(ckpt))
                # name = '54.png'
                im = cv2.imread('./dataset/test/input/' + name)[:, :, ::-1]
                # print(im)
                im_net = im / 255
                label = cv2.imread('./dataset/test/label/' + name)[:, :, ::-1]
                label_net = label / 255
                    # print(im)

                out, p, s, m = sess.run([y, psnr, ssim, mse], feed_dict={x: [im_net], y_: [label_net]})

                # print(label.dtype)
                # print(out)
                out_im = out[0] * 255
                # print(out_im)0
                # out_im = out_im.astype(np.uint8)[:, :, ::-1]
                out_im = out_im.astype(np.uint8)
                print(out_im)
                # cv2.imshow('in', im[:, :, ::-1])
                # cv2.imshow('label', label[:, :, ::-1])
                # cv2.imshow('out', out_im)
                # cv2.waitKey(0)

                sm.imshow(im)
                sm.imshow(label)
                sm.imshow(out_im)

                print(im[0, 0])
                print(out_im[0, 0])
                p1 = cp.fun(im, out_im, m=255.0, BGR=True)
                s1 = compare_ssim(im_net, out[0], multichannel=True, full=False)
                print('tensorflow:', p, s)
                print('ski:', p1, s1)
                if save_result:
                    cv2.imwrite('result.png', out_im)
                # cv2.waitKey(0)

if __name__ == '__main__':
    main(parse_arguments_test(sys.argv[1:]))
