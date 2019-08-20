import tensorflow.contrib.slim as slim
import tensorflow as tf


def model(inputs, is_training, regular=None, use_detail=False, use_se=False):

    # if use_detail:
    #     base = guided_filter(inputs, inputs, 15, 1, nhwc=True)  # using guided filter for obtaining base layer
    #     net = inputs - base  # detail layer
    # else:
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


def model_my(inputs, is_training, regular=0.0001):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
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


def se_block(x, ratio, name):
    out_dim = x.shape[-1]
    print("se_block: out_dim = {}, ratio = {}".format(out_dim, ratio))
    with tf.name_scope(name=name):
        squeeze = tf.reduce_mean(x, axis=[1, 2])
        excitation = slim.fully_connected(squeeze, out_dim / ratio, activation_fn=tf.nn.relu)
        excitation - slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.sigmoid)
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = x * excitation
    return scale



def roi(inputs, batch_norm_params):
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,):
        net = slim.conv2d(inputs, 128, 9, scope='conv1')
        net = tf.concat([net, inputs], axis=-1)
        net = se_block(net, ratio=8, name='se1')
        net = slim.conv2d(net, 128, 1, scope='conv2')
        net = tf.concat([net, inputs], axis=-1)
        net = se_block(net, ratio=8, name='se2')
        net = slim.conv2d(net, 64, 5, scope='conv3')
        net = tf.concat([net, inputs], axis=-1)
        net = se_block(net, ratio=4, name='se3')
        net = slim.conv2d(net, 64, 3, scope='conv4')
        net = tf.concat([net, inputs], axis=-1)
        net = se_block(net, ratio=4, name='se4')
    net = slim.conv2d(net, 3, 3, scope='conv5', activation_fn=None)
    return net


def choice(inputs, tag, is_training):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
    }
    if tag == 1 or tag == 0:
        return roi(inputs, batch_norm_params)
