import tensorflow as tf


def Global_Avg_Pooling(x):
    return tf.reduce_mean(x, axis=[1, 2], name="Global_Avg_Pooling")


def ReLU(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Fully_Connected(x, units, layer_name="fully_connected"):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def SE_Block(x, out_dim, ratio, layer_name="SE_Block"):
    print("se_block: out_dim = {}, ratio = {}".format(out_dim, ratio))
    with tf.name_scope(layer_name):
        squeeze = Global_Avg_Pooling(x)
        excitation = Fully_Connected(squeeze, units=out_dim / ratio, layer_name=layer_name + "_fc1")
        excitation = ReLU(excitation)
        excitation = Fully_Connected(excitation, units=out_dim, layer_name=layer_name + "_fc2")
        excitation = Sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = x * excitation
        return scale
