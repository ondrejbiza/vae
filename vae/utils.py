import tensorflow as tf


def get_weight_regularizer(weight_decay):
    """
    Get L2 weight regularizer.
    :param weight_decay:    Weight decay.
    :return:                Regularizer object.
    """

    return tf.contrib.layers.l2_regularizer(weight_decay)


def hardtanh(x):

    return tf.minimum(tf.maximum(-1.0, x), 1.0)
