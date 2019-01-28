import tensorflow as tf
from vae import VAE

((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

model = VAE([28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001)

model.start_session()
model.stop_session()
