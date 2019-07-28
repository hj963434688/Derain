import numpy as np
import data_process as d
import tensorflow as tf
import warnings
from PIL import  Image
import cv2
import scipy.misc as sm
import data_process as dp
warnings.filterwarnings("ignore")


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def run_mse():

    img_generator = d.get_batch(num_workers=8,
                                  input_size=33,
                                  batch_size=16)

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    y = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_labels')
    # x = mean_image_subtraction(x)
    # y = mean_image_subtraction(y)
    # x = tf.image.per_image_standardization(x)
    # y = tf.image.per_image_standardization(y)

    a = tf.square(x - y)
    # print(x.shape)
    # x = tf.reshape(x, [3, -1])
    # y = tf.reshape(y, [3, -1])
    #
    mse = tf.reduce_mean(tf.square(x - y))

    with tf.Session() as sess:
        while True:
            data = next(img_generator)
            # b = sess.run(a, feed_dict={x: data[0], y: data[1]})
            # print(b)
            # print(data[0].shape)
            m = sess.run(mse, feed_dict={x: data[0], y: data[1]})
            print(m)


def tf_group_tuple():
    # demo1
    # when range(500)two results randomly occur:499 and 500
    # when range(3000)two results randomly occur:2999 and 3000

    with tf.name_scope('initial'):
        a = tf.Variable(0, dtype=tf.float32)
        b = tf.Variable(0, dtype=tf.float32)

    # update
    update1 = tf.assign_add(a, 1)
    update2 = tf.assign_sub(a, 1)
    update3 = tf.assign(b, a)

    update = tf.group(update1, update2, update3)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for _ in range(30000):
            _ = sess.run(update)
            print(sess.run(b))


def png_jpg():
    # im1 = cv2.imread('./dataset/train/input/2.jpg')[:, :, ::-1]
    # im2 = cv2.imread('./dataset/train/input/2.png')[:, :, ::-1]
    im3 = sm.imread('./dataset/train/input/2.png', mode='RGB')
    cv2.imshow('im', im3[:, :, ::-1])
    cv2.waitKey(0)
    # print(im2.shape)
    print(im3.shape)
    size = 100
    # print(im2[0][1])1
    # print(im3[0][1])
    h, w, _ = im3.shape
    h_n = int(h // size)
    w_n = int(w // size)
    imglist = []
    for i in range(h_n):
        for j in range(w_n):
            imglist.append(im3[i*size:(i+1)*size, j*size:(j+1)*size, :])
    for im in imglist:
        cv2.imshow('im', im[:, :, ::-1])
        cv2.waitKey(0)


def get_val():
    input_list, label_list = dp.get_images(dp.train_path)
    print(len(input_list))
    print(input_list)
    print(label_list)


if __name__ == '__main__':
    # run_mse()
    # # tf_group_tuple()
    # png_jpg()
    # i = np.arange(0, 4)
    # print(i)
    get_val()