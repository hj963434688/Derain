import tensorflow.contrib.slim as slim
import tensorflow as tf
from nets.roi import roi


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


def choice(inputs, tag, is_training):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
    }
    if tag == 1 or tag == 0:
        return roi(inputs, batch_norm_params)
