from enum import Enum
from operator import mul
from functools import reduce
import tensorflow as tf
import utils


class VAE:

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    def __init__(self, input_shape, encoder_neurons, decoder_neurons, latent_space_size, loss_type,
                 weight_decay, learning_rate):

        assert loss_type in self.LossType

        self.input_shape = input_shape
        self.flat_input_shape = reduce(mul, self.input_shape, 1)
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_space_size = latent_space_size
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.build_placeholders()
        self.build_network()
        self.build_training()

        self.input_pl = None
        self.input_flat_t = None
        self.mu_t = None
        self.log_sd_t = None
        self.sd_t = None
        self.var_t = None
        self.mean_sq_t = None
        self.kl_divergence_t = None
        self.kl_loss_t = None
        self.noise_t = None
        self.sd_noise_t = None
        self.sample_t = None
        self.output_t = None
        self.output_loss_t = None
        self.loss_t = None
        self.step_op = None
        self.session = None

    def build_placeholders(self):

        self.input_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="input_pl")

    def build_network(self):

        self.input_flat_t = tf.reshape(self.input_pl, shape=(tf.shape(self.input_pl)[0], self.flat_input_shape))

        # encoder
        x = self.input_flat_t

        for neurons in self.encoder_neurons:
            x = tf.layers.dense(
                x, neurons, activation=tf.nn.relu, kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
            )

        # middle
        self.mu_t = tf.layers.dense(
            x, self.latent_space_size, activation=None,
            kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
        )
        self.log_sd_t = tf.layers.dense(
            x, self.latent_space_size, activation=None,
            kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
        )

        self.sd_t = tf.exp(self.log_sd_t)
        self.var_t = self.sd_t * self.sd_t
        self.mean_sq_t = self.mu_t * self.mu_t

        self.kl_divergence_t = 0.5 * self.mean_sq_t + 0.5 * self.var_t - 1.0 * self.log_sd_t - 0.5

        self.noise_t = tf.random.normal(shape=(tf.shape(self.mu_t)[0], self.latent_space_size), mean=0, stddev=1.0)

        self.sd_noise_t = self.noise_t * self.sd_t
        self.sample_t = self.mu_t + self.sd_noise_t

        # decoder
        x = self.sample_t

        for idx, neurons in enumerate(self.decoder_neurons):
            x = tf.layers.dense(
                x, neurons, activation=tf.nn.relu if idx != len(self.decoder_neurons) - 1 else None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
            )

        self.output_t = x

    def build_training(self):

        if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
            self.output_loss_t = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.input_flat_t,
                                                                 logits=self.output_t)
        else:
            self.output_loss_t = tf.losses.mean_squared_error(labels=self.input_flat_t, predictions=self.output_t)

        self.kl_loss_t = tf.reduce_mean(tf.reduce_sum(self.kl_divergence_t, axis=1))

        self.loss_t = self.output_loss_t + self.kl_loss_t

        self.step_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def start_session(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()
