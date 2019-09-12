import numpy as np
import matplotlib.pyplot as plt
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


def many_multivariate_normals_log_pdf(x, mu, var, logvar):

    x = x[:, tf.newaxis, :]
    mu = mu[tf.newaxis, :, :]
    var = var[tf.newaxis, :, :]
    logvar = logvar[tf.newaxis, :, :]

    term1 = - (mu.shape[2].value / 2) * np.log(2 * np.pi)
    term2 = - (1 / 2) * tf.reduce_sum(logvar, axis=2)
    term3 = - (1 / 2) * tf.reduce_sum(tf.square(x - mu) / var, axis=2)

    return term1 + term2 + term3


def plot_square(images):

    n = int(np.sqrt(len(images)))

    _, axes = plt.subplots(nrows=5, ncols=5)

    for i in range(len(images)):
        axis = axes[i // n, i % n]

        axis.imshow(images[i], vmin=0, vmax=1, cmap="gray")
        axis.axis("off")
