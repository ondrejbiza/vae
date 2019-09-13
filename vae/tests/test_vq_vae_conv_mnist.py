import os
import logging
import unittest
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from ..vq_vae_conv import VQ_VAE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestVQVAE(unittest.TestCase):

    LATENT_SIZE = 11
    NUM_EMBEDDINGS = 13
    EMBEDDING_SIZE = 17

    def tearDown(self):
        tf.reset_default_graph()

    def get_tensor_shape(self, tensor):

        shape = tensor.shape
        shape = tuple([item.value for item in shape])

        return shape

    def test_shapes(self):

        model = VQ_VAE(
            [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4],
            [2, 2, 2, 1], self.LATENT_SIZE, self.NUM_EMBEDDINGS, self.EMBEDDING_SIZE,
            VQ_VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001, 0.4, 0.25
        )
        model.build_all()

        self.assertEqual(self.get_tensor_shape(model.pred_embeds), (None, self.LATENT_SIZE, self.EMBEDDING_SIZE))
        self.assertEqual(
            self.get_tensor_shape(model.diff), (None, self.LATENT_SIZE, self.NUM_EMBEDDINGS, self.EMBEDDING_SIZE)
        )
        self.assertEqual(self.get_tensor_shape(model.norm), (None, self.LATENT_SIZE, self.NUM_EMBEDDINGS))
        self.assertEqual(self.get_tensor_shape(model.classes), (None, self.LATENT_SIZE))
        self.assertEqual(self.get_tensor_shape(model.flat_classes), (None,))
        self.assertEqual(self.get_tensor_shape(model.vector_of_collected_embeds), (None, self.EMBEDDING_SIZE))
        self.assertEqual(self.get_tensor_shape(model.collected_embeds), (None, self.LATENT_SIZE, self.EMBEDDING_SIZE))
        self.assertEqual(self.get_tensor_shape(
            model.collected_embeds_fake_grads), (None, self.LATENT_SIZE, self.EMBEDDING_SIZE)
        )
        self.assertEqual(
            self.get_tensor_shape(model.flat_collected_embeds_fake_grads),
            (None, self.LATENT_SIZE * self.EMBEDDING_SIZE)
        )

    def test_build_and_start_session(self):

        model = VQ_VAE(
            [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4],
            [2, 2, 2, 1], 11, 13, 17, VQ_VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001, 0.4, 0.25
        )
        model.build_all()
        model.start_session()
        model.stop_session()

    def test_predict(self):

        model = VQ_VAE(
            [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4],
            [2, 2, 2, 1], 11, 13, 17, VQ_VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001, 0.4, 0.25
        )
        model.build_all()
        model.start_session()

        outputs, classes = model.predict(2)

        self.assertEqual(outputs.shape, (2, 28, 28))
        self.assertEqual(int(np.sum(np.isnan(outputs))), 0)

        model.stop_session()

    def test_train(self):

        model = VQ_VAE(
            [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4],
            [2, 2, 2, 1], 11, 13, 17, VQ_VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001, 0.4, 0.25
        )
        model.build_all()
        model.start_session()

        model.train(np.random.uniform(0, 1, (2, 28, 28)))

        model.stop_session()
