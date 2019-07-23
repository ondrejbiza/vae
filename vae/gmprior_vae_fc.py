from enum import Enum
import numpy as np
import tensorflow as tf
from . import utils
from .model import Model


class GMPRIOR_VAE(Model):

    MODEL_NAMESPACE = "model"
    TRAINING_NAMESPACE = "training"

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    def __init__(self, input_shape, encoder_neurons, decoder_neurons, latent_space_size, loss_type, weight_decay,
                 learning_rate, num_components, beta1=1.0, beta2=1.0, fix_cudnn=False):

        super(GMPRIOR_VAE, self).__init__(fix_cudnn=fix_cudnn)

        assert loss_type in self.LossType

        self.input_shape = input_shape
        self.flat_input_shape = int(np.prod(self.input_shape))
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_space_size = latent_space_size
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.num_components = num_components
        self.beta1 = beta1
        self.beta2 = beta2

        self.input_pl = None
        self.input_flat_t = None
        self.mu_t = None
        self.log_var_t = None
        self.sd_t = None
        self.var_t = None
        self.mean_sq_t = None
        self.mixtures_mu_v = None
        self.mixtures_logvar_v = None
        self.mixtures_var_t = None
        self.sample_probs_t = None
        self.encoder_entropy_t = None
        self.pseudo_expectation_t = None
        self.entropy_loss_t = None
        self.prior_loss_t = None
        self.reg_loss_t = None
        self.noise_t = None
        self.sd_noise_t = None
        self.sample_t = None
        self.logits_t = None
        self.output_t = None
        self.output_loss_t = None
        self.loss_t = None
        self.step_op = None
        self.session = None

        self.build_placeholders()
        self.build_network()
        self.build_training()

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MODEL_NAMESPACE)
        )

    def predict(self, num_samples):

        mixtures_mu, mixtures_var = self.session.run([self.mixtures_mu_v, self.mixtures_var_t])
        assignments = np.random.randint(0, self.num_components, size=num_samples)

        flat_outputs = self.session.run(self.output_t, feed_dict={
            self.mu_t: mixtures_mu[assignments],
            self.sd_t: np.sqrt(mixtures_var[assignments])

        })

        outputs = np.reshape(flat_outputs, (num_samples, *self.input_shape))

        return outputs, mixtures_mu

    def train(self, samples):

        _, loss, output_loss, entropy_loss, prior_loss, reg_loss = self.session.run(
            [self.step_op, self.loss_t, self.output_loss_t, self.entropy_loss_t, self.prior_loss_t, self.reg_loss_t],
            feed_dict={
                self.input_pl: samples
            }
        )

        return loss, output_loss, entropy_loss, prior_loss, reg_loss

    def build_placeholders(self):

        self.input_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="input_pl")

    def build_encoder(self, input_t, reuse=False):

        x = input_t

        with tf.variable_scope("encoder", reuse=reuse):

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("layer{:d}".format(idx + 1), reuse=reuse):
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

    def build_network(self):

        with tf.variable_scope(self.MODEL_NAMESPACE):

            self.input_flat_t = tf.reshape(self.input_pl, shape=(tf.shape(self.input_pl)[0], self.flat_input_shape))

            # encoder
            self.mu_t, self.log_var_t = self.build_encoder(self.input_flat_t)

            # mixtures
            self.mixtures_mu_v = tf.get_variable(
                "mixtures_mu", initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32
                ), shape=(self.num_components, self.latent_space_size)
            )
            self.mixtures_logvar_v = tf.get_variable(
                "mixtures_var", initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32
                ), shape=(self.num_components, self.latent_space_size)
            )

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
                self.mixtures_var_t = tf.exp(self.mixtures_logvar_v)

                self.sample_probs_t = utils.many_multivariate_normals_log_pdf(
                    self.sample_t, self.mixtures_mu_v, self.mixtures_logvar_v, self.mixtures_var_t
                )
                self.sample_probs_t -= np.log(self.num_components)

                d_max = tf.reduce_max(self.sample_probs_t, axis=1)
                self.pseudo_expectation_t = d_max + tf.log(
                    tf.reduce_sum(tf.exp(self.sample_probs_t - d_max[:, tf.newaxis]), axis=1)
                )

            # decoder
            x = self.sample_t

            with tf.variable_scope("decoder"):
                for idx, neurons in enumerate(self.decoder_neurons):
                    with tf.variable_scope("layer{:d}".format(idx + 1)):
                        x = tf.layers.dense(
                            x, neurons, activation=tf.nn.relu if idx != len(self.decoder_neurons) - 1 else None,
                            kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
                        )

            self.logits_t = x

            if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
                self.output_t = tf.nn.sigmoid(self.logits_t)
            else:
                self.output_t = self.logits_t

    def build_training(self):

        with tf.variable_scope(self.TRAINING_NAMESPACE):

            if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
                self.output_loss_t = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.losses.sigmoid_cross_entropy(
                            multi_class_labels=self.input_flat_t, logits=self.logits_t,
                            reduction=tf.losses.Reduction.NONE
                        ),
                        axis=1
                    ),
                    axis=0
                )
            else:
                self.output_loss_t = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.losses.mean_squared_error(
                            labels=self.input_flat_t, predictions=self.logits_t,
                            reduction=tf.losses.Reduction.NONE
                        ),
                        axis=1
                    ),
                    axis=0
                )

            self.entropy_loss_t = - tf.reduce_mean(self.encoder_entropy_t, axis=0)

            self.prior_loss_t = - tf.reduce_mean(self.pseudo_expectation_t, axis=0)

            self.reg_loss_t = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss_t = self.output_loss_t + self.beta1 * self.entropy_loss_t + self.beta2 * self.prior_loss_t + \
                self.reg_loss_t

            self.step_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_t)
