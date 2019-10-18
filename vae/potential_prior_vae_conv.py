from enum import Enum
import numpy as np
import tensorflow as tf
from . import utils
from .model import Model


class POTENTIAL_PRIOR_VAE(Model):

    MODEL_NAMESPACE = "model"
    TRAINING_NAMESPACE = "training"

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    class PriorType(Enum):

        EXP_TANH = 1
        EXP_COSINE_SIM = 2

    def __init__(self, input_shape, encoder_filters, encoder_filter_sizes, encoder_strides, encoder_neurons,
                 decoder_neurons, decoder_filters, decoder_filter_sizes, decoder_strides, latent_space_size, loss_type,
                 prior_type, weight_decay, learning_rate, num_components, beta1=1.0, beta2=1.0, tau=1.0,
                 fix_cudnn=False):

        super(POTENTIAL_PRIOR_VAE, self).__init__(fix_cudnn=fix_cudnn)

        assert loss_type in self.LossType

        self.input_shape = input_shape
        self.flat_input_shape = int(np.prod(self.input_shape))
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.decoder_filters = decoder_filters
        self.decoder_filter_sizes = decoder_filter_sizes
        self.decoder_strides = decoder_strides
        self.latent_space_size = latent_space_size
        self.loss_type = loss_type
        self.prior_type = prior_type
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.num_components = num_components
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        self.input_pl = None
        self.input_flat_t = None
        self.mu_t = None
        self.log_var_t = None
        self.sd_t = None
        self.var_t = None
        self.mean_sq_t = None
        self.mixtures_mu_v = None
        self.encoder_entropy_t = None
        self.pseudo_expectation_t = None
        self.sample_potential_t = None
        self.entropy_loss_t = None
        self.prior_loss_t = None
        self.reg_loss_t = None
        self.noise_t = None
        self.sd_noise_t = None
        self.sample_t = None
        self.logits_t = None
        self.flat_logits_t = None
        self.output_t = None
        self.output_loss_t = None
        self.loss_t = None
        self.step_op = None

        self.build_placeholders()
        self.build_network()
        self.build_training()

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MODEL_NAMESPACE)
        )

    def predict(self, num_samples):

        mixtures_mu = self.get_mixtures()
        assignments = np.random.randint(0, self.num_components, size=num_samples)

        flat_outputs = self.session.run(self.output_t, feed_dict={
            self.mu_t: mixtures_mu[assignments],
            self.var_t: np.ones_like(mixtures_mu[assignments]) * 0.1
        })

        return flat_outputs, mixtures_mu

    def predict_x_from_mixture(self, mu, var, num_samples):

        flat_outputs = self.session.run(self.output_t, feed_dict={
            self.mu_t: np.tile(mu[np.newaxis, :], [num_samples, 1]),
            self.var_t: np.tile(var[np.newaxis, :], [num_samples, 1])
        })

        return flat_outputs

    def train(self, samples):

        _, loss, output_loss, entropy_loss, prior_loss, reg_loss, sp, n1, n2 = self.session.run(
            [self.step_op, self.loss_t, self.output_loss_t, self.entropy_loss_t, self.prior_loss_t, self.reg_loss_t,
             self.sample_potential_t, self.mu_norms_t, self.mixtures_mu_norms_t],
            feed_dict={
                self.input_pl: samples
            }
        )

        return loss, output_loss, entropy_loss, prior_loss, reg_loss, n1, n2

    def get_mixtures(self):

        return np.transpose(self.session.run(self.mixtures_mu_v))

    def build_placeholders(self):

        self.input_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="input_pl")
        self.input_flat_t = tf.reshape(self.input_pl, shape=(tf.shape(self.input_pl)[0], self.flat_input_shape))

    def build_encoder(self, input_t, reuse=False):

        x = tf.expand_dims(input_t, axis=-1)

        with tf.variable_scope("encoder", reuse=reuse):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                    )

            x = tf.layers.flatten(x)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                    )

            with tf.variable_scope("heads", reuse=reuse):

                mu_t = tf.layers.dense(
                    x, self.latent_space_size, activation=None,
                    kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                )
                log_var_t = tf.layers.dense(
                    x, self.latent_space_size, activation=None,
                    kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                )

        return mu_t, log_var_t

    def build_decoder(self, input_t, reuse=False):

        with tf.variable_scope("decoder", reuse=reuse):

            x = input_t

            for idx, neurons in enumerate(self.decoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                    )

            x = x[:, tf.newaxis, tf.newaxis, :]

            for idx in range(len(self.decoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d_transpose(
                        x, self.decoder_filters[idx], self.decoder_filter_sizes[idx], self.decoder_strides[idx],
                        padding="VALID", activation=tf.nn.relu if idx != len(self.decoder_filters) - 1 else None,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                    )

        logits_t = x
        flat_logits_t = tf.layers.flatten(x)

        if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
            output_t = tf.nn.sigmoid(logits_t)
        else:
            output_t = logits_t

        return logits_t, flat_logits_t, output_t

    def build_network(self):

        with tf.variable_scope(self.MODEL_NAMESPACE):

            self.input_flat_t = tf.reshape(self.input_pl, shape=(tf.shape(self.input_pl)[0], self.flat_input_shape))

            # encoder
            self.mu_t, self.log_var_t = self.build_encoder(self.input_pl)

            # mixtures
            self.mixtures_mu_v = tf.get_variable(
                "mixtures_mu", shape=(self.latent_space_size, self.num_components)
            )

            self.mu_norms_t = tf.reduce_mean(tf.square(tf.norm(self.mu_t, ord=2, axis=1)))
            self.mixtures_mu_norms_t = tf.reduce_mean(tf.square(tf.norm(self.mixtures_mu_v, ord=2, axis=0)))

            # middle
            with tf.variable_scope("middle"):

                # sample encoder
                self.var_t = tf.exp(self.log_var_t)
                self.sd_t = tf.sqrt(self.var_t)
                self.mean_sq_t = tf.square(self.mu_t)

                self.noise_t = tf.random_normal(
                    shape=(tf.shape(self.mu_t)[0], self.latent_space_size), mean=0, stddev=1.0
                )

                self.sd_noise_t = self.noise_t * self.sd_t
                self.sample_t = self.mu_t + self.sd_noise_t

                # calculate encoder entropy
                self.encoder_entropy_t = (1 / 2) * tf.reduce_sum(self.log_var_t, axis=1) + \
                    (self.latent_space_size / 2) * (1 + np.log(2 * np.pi))

                # calculate prior expectation
                if self.prior_type is self.PriorType.EXP_TANH:
                    self.sample_potential_t = self.prior_exp_tanh(self.sample_t, self.mixtures_mu_v)
                else:
                    self.sample_potential_t = self.prior_exp_cosine_similarity(self.sample_t, self.mixtures_mu_v)

                self.pseudo_expectation_t = tf.log(tf.reduce_sum(self.sample_potential_t, axis=1))

            # decoder
            self.logits_t, self.flat_logits_t, self.output_t = self.build_decoder(self.sample_t)

            if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
                self.output_t = tf.nn.sigmoid(self.logits_t)
            else:
                self.output_t = self.logits_t

    def build_training(self):

        with tf.variable_scope(self.TRAINING_NAMESPACE):

            if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
                self.full_output_loss_t = tf.reduce_sum(
                    tf.losses.sigmoid_cross_entropy(
                        multi_class_labels=self.input_flat_t, logits=self.flat_logits_t,
                        reduction=tf.losses.Reduction.NONE
                    ),
                    axis=1
                )
            else:
                self.full_output_loss_t = tf.reduce_sum(
                    tf.losses.mean_squared_error(
                        labels=self.input_flat_t, predictions=self.flat_logits_t,
                        reduction=tf.losses.Reduction.NONE
                    ),
                    axis=1
                )

            self.output_loss_t = tf.reduce_mean(self.full_output_loss_t, axis=0)

            self.entropy_loss_t = - tf.reduce_mean(self.encoder_entropy_t, axis=0)

            self.prior_loss_t = - tf.reduce_mean(self.pseudo_expectation_t, axis=0)

            self.reg_loss_t = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss_t = self.output_loss_t + self.beta1 * self.entropy_loss_t + self.beta2 * self.prior_loss_t + \
                self.reg_loss_t

            self.step_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_t)

    def prior_exp_tanh(self, x, mu):

        output = tf.matmul(x, mu)
        output = tf.nn.tanh(output)
        output = tf.exp(output / self.tau)

        return output

    def prior_exp_cosine_similarity(self, x, mu):

        norm_x = tf.norm(x, ord=2, axis=1)
        norm_mu = tf.norm(mu, ord=2, axis=0)

        x_div = x / norm_x[:, tf.newaxis]
        mu_div = mu / norm_mu[tf.newaxis, :]

        output = tf.matmul(x_div, mu_div)
        output = tf.exp(output / self.tau)

        return output
