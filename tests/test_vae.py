import unittest
import tensorflow as tf
from vae import VAE


class TestVAE(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_build_and_start_session(self):

        model = VAE(
            [28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001
        )
        model.start_session()
        model.stop_session()
