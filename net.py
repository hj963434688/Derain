import tensorflow.contrib.slim as slim
import tensorflow as tf
from CVPR17_training_code.GuidedFilter import guided_filter


def model(inputs, is_training, regular=0.0001, use_detail=False, use_se=False):

    if use_detail:
        base = guided_filter(inputs, inputs, 15, 1, nhwc=True)  # using guided filter for obtaining base layer
        net = inputs - base  # detail layer
    else:
        net = inputs

    batch_norm_params = {
        'decay': 0.999,
        'epsilon': 1e-4,
        'scale': True,
        'is_training': is_training
    }
    ratio = 8
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=None):
        net = slim.conv2d(net, 32, 3, scope='conv1')
        net1 = slim.conv2d(net, 32, 3, rate=1, scope='conv2/r1')
        net1 = slim.conv2d(net1, 32, 3, rate=1, scope='conv3/r1')
        net2 = slim.conv2d(net, 32, 3, rate=2, scope='conv2/r2')
        net2 = slim.conv2d(net2, 32, 3, rate=2, scope='conv3/r2')
        net3 = slim.conv2d(net, 32, 3, rate=3, scope='conv2/r3')
        net3 = slim.conv2d(net3, 32, 3, rate=3, scope='conv3/r3')
        net = tf.concat([net1, net2, net3], axis=-1)

        if use_se:
            squeeze = tf.reduce_mean(net, axis=[1, 2], name='global_pooling')
            excitation = slim.fully_connected(squeeze, 96 // ratio, activation_fn=tf.nn.relu, scope='se/fc1')
            excitation = slim.fully_connected(excitation, 96, activation_fn=tf.nn.sigmoid, scope='se/fc2')
            excitation = tf.reshape(excitation, [-1, 1, 1, 96])
            net = net * excitation

        net = slim.conv2d(net, 32, 3, scope='conv4')

    net = slim.conv2d(net, 3, 3, scope='conv5', activation_fn=None)
    if use_detail:
        net = tf.add(inputs, net)

    return net


def model_roi(inputs, is_training, regular=0.0001):
    batch_norm_params = {
        'decay': 0.999,
        'epsilon': 1e-4,
        'scale': True,
        'is_training': is_training
    }
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=None):
        net = slim.conv2d(inputs, 128, 9, scope='conv1')
        net = tf.concat([net, inputs], axis=-1)
        net = slim.conv2d(net, 128, 1, scope='conv2')
        net = tf.concat([net, inputs], axis=-1)
        net = slim.conv2d(net, 64, 5, scope='conv3')
        net = tf.concat([net, inputs], axis=-1)
        net = slim.conv2d(net, 64, 3, scope='conv4')
        net = tf.concat([net, inputs], axis=-1)

    net = slim.conv2d(net, 3, 3, scope='conv5', activation_fn=None)
    return net


def model_my(inputs, is_training, regular=0.0001):
    batch_norm_params = {
        'decay': 0.999,
        'epsilon': 1e-4,
        'scale': True,
        'is_training': is_training
    }
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=None):
        net = slim.conv2d(inputs, 128, 9, scope='conv1')
        net = tf.concat([inputs, net], axis=-1)
        pr1 = slim.conv2d(net, 32, 1, scope='pr1/1')
        pr1 = slim.conv2d(pr1, 3, 3, scope='pr1/2')
        net = slim.conv2d(net, 128, 1, scope='conv2')
        net = tf.concat([inputs, net], axis=-1)
        pr2 = slim.conv2d(net, 32, 1, scope='pr2/1')
        pr2 = slim.conv2d(pr2, 3, 3, scope='pr2/2')
        net = slim.conv2d(net, 64, 5, scope='conv3')
        net = tf.concat([inputs, net], axis=-1)
        pr3 = slim.conv2d(net, 32, 1, scope='pr3/1')
        pr3 = slim.conv2d(pr3, 3, 3, scope='pr3/2')
        net = slim.conv2d(net, 64, 3, scope='conv4')
        net = tf.concat([inputs, net], axis=-1)
        pr4 = slim.conv2d(net, 32, 1, scope='pr4/1')
        pr4 = slim.conv2d(pr4, 3, 3, scope='pr4/2')
        net = tf.concat([pr1, pr2, pr3, pr4], axis=-1)

    net = slim.conv2d(net, 3, 1, scope='conv5', activation_fn=None)
    return net
