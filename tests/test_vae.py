import os
import unittest
import numpy as np
import tensorflow as tf
from vae import VAE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestVAE(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_build_and_start_session(self):

        model = VAE(
            [28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001
        )
        model.start_session()
        model.stop_session()

    def test_predict(self):

        model = VAE(
            [28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001
        )
        model.start_session()

        outputs = model.predict(2)

        self.assertEqual(outputs.shape, (2, 28, 28))
        self.assertEqual(int(np.sum(np.isnan(outputs))), 0)

        model.stop_session()

    def test_train(self):

        model = VAE(
            [28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001
        )
        model.start_session()

        model.train(np.random.uniform(0, 1, (2, 28, 28)))

        model.stop_session()
