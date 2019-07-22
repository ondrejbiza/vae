from enum import Enum
import numpy as np
import tensorflow as tf
from . import utils
from .model import Model


class VAMPPRIOR_VAE(Model):

    MODEL_NAMESPACE = "model"
    TRAINING_NAMESPACE = "training"

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    def __init__(self, input_shape, encoder_filters, encoder_filter_sizes, encoder_strides, encoder_neurons,
                 decoder_neurons, decoder_filters, decoder_filter_sizes, decoder_strides, latent_space_size, loss_type,
                 weight_decay, learning_rate, num_pseudo_inputs, pseudo_inputs_activation=None, beta1=1.0, beta2=1.0,
                 fix_cudnn=False):

        super(VAMPPRIOR_VAE, self).__init__(fix_cudnn=fix_cudnn)

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
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.num_pseudo_inputs = num_pseudo_inputs
        self.pseudo_inputs_activation = pseudo_inputs_activation
        self.beta1 = beta1
        self.beta2 = beta2

        self.input_pl = None
        self.input_flat_t = None
        self.mu_t = None
        self.log_var_t = None
        self.sd_t = None
        self.var_t = None
        self.mean_sq_t = None
        self.pseudo_inputs_t = None
        self.pseudo_mu_t = None
        self.pseudo_var_t = None
        self.reg_loss_t = None
        self.noise_t = None
        self.sd_noise_t = None
        self.sample_t = None
        self.logits_t = None
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

        pseudo_mu, pseudo_var = self.session.run([self.pseudo_mu_t, self.pseudo_var_t])
        assignments = np.random.randint(0, self.num_pseudo_inputs, size=num_samples)

        flat_outputs = self.session.run(self.output_t, feed_dict={
            self.mu_t: pseudo_mu[assignments],
            self.sd_t: pseudo_var[assignments]

        })

        return flat_outputs, pseudo_mu

    def train(self, samples):

        _, loss, output_loss, entropy_loss, prior_loss, reg_loss, sp = self.session.run(
            [self.step_op, self.loss_t, self.output_loss_t, self.entropy_loss_t, self.prior_loss_t, self.reg_loss_t,
             self.sample_probs_t],
            feed_dict={
                self.input_pl: samples
            }
        )

        return loss, output_loss, entropy_loss, prior_loss, reg_loss

    def get_pseudo_inputs(self):

        return self.session.run(self.pseudo_inputs_t)

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

            # pseudo inputs
            self.pseudo_inputs_t = tf.get_variable(
                "pseudo_inputs", initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32
                ), shape=(self.num_pseudo_inputs, *self.input_shape)
            )
            if self.pseudo_inputs_activation is not None:
                self.pseudo_inputs_t = self.pseudo_inputs_activation(self.pseudo_inputs_t)

            self.pseudo_mu_t, self.pseudo_logvar_t = self.build_encoder(self.pseudo_inputs_t, reuse=True)

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
                self.pseudo_var_t = tf.exp(self.pseudo_logvar_t)
                self.sample_probs_t = utils.many_multivariate_normals_log_pdf(
                    self.sample_t, self.pseudo_mu_t, self.pseudo_var_t, self.pseudo_logvar_t
                )
                self.sample_probs_t -= np.log(self.num_pseudo_inputs)

                d_max = tf.reduce_max(self.sample_probs_t, axis=1)
                self.pseudo_expectation_t = d_max + tf.log(
                    tf.reduce_sum(tf.exp(self.sample_probs_t - d_max[:, tf.newaxis]), axis=1)
                )

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
