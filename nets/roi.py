from nets import *


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


def roi_se(inputs, batch_norm_params):
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