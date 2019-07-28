import tensorflow as tf
import os
import data_process
import numpy as np
import time
import cv2
import scipy.misc as sm
import net
import train as tr
import sys
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import compu_psnr as cp
import compu_ssim as cs

output_path = './dataset/test/result/'
test_path = './dataset/test'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

save_result = False


def test():

    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != 17:
            raise
    with tf.get_default_graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        y_ = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='label_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # y = net.model(x, False, use_se=True)
        y = net.model_roi(x, False)
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
                psnr_list = []
                ssim_list = []
                start = time.time()
                files = data_process.get_images(test_path)
                for i in range(len(files[0])):
                    # im = cv2.imread('./dataset/test/input/133.png')[:, :, ::-1] / 255
                    # label = cv2.imread('./dataset/test/label/133.png')[:, :, ::-1] / 255
                    # print(im)

                    im = sm.imread(files[0][i], mode='RGB')
                    im_net = im / 255
                    label = sm.imread(files[1][i], mode='RGB')
                    label_net = label / 255

                    out, p, s, m = sess.run([y, psnr, ssim, mse], feed_dict={x: [im_net], y_: [label_net]})

                    out_net = out[0]
                    out = out_net * 255

                    out_int = out.astype(np.uint8)
                    # out_45 = np.rint(out)
                    #sint = cp.funssim(label, out_int, )
                    #s45 = cp.funssim(label, out_int)
                    #sx = cp.funssimx(label, out_int)
                    p1 = cp.fun(label, out_int, m=255.0, BGR=False)
                    # p2 = cp.fun(im, label, m=255.0, BGR=False)
                    # p4 = cp.psnr1(label, out)
                    # p5 = cp.psnr2(label, out)

                    print(files[0][i], p, s, p1)
                    psnr_list.append(p1)
                    ssim_list.append(s)
                print(psnr_list)
                print(ssim_list)
                print(sum(psnr_list) / len(psnr_list))
                print(sum(ssim_list) / len(ssim_list))
                print(time.time() - start)
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


def main(a):
    # val()
    if a[0] == 'all':
        test()
    else:
        test_one(a[0] + '.png')


if __name__ == '__main__':
    try:
        tf.app.run(main=main, argv=[sys.argv[1]])
    except:
        test()
    # test_ing()
    # test_one('66.png')