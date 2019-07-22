from enum import Enum
import numpy as np
import tensorflow as tf
from . import utils
from .model import Model


class VAE(Model):

    MODEL_NAMESPACE = "model"
    TRAINING_NAMESPACE = "training"

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    def __init__(self, input_shape, encoder_filters, encoder_filter_sizes, encoder_strides, encoder_neurons,
                 decoder_neurons, decoder_filters, decoder_filter_sizes, decoder_strides, latent_space_size, loss_type,
                 weight_decay, learning_rate, disable_kl_loss=False, fix_cudnn=False):

        super(VAE, self).__init__(fix_cudnn=fix_cudnn)

        assert loss_type in self.LossType
        assert len(encoder_filters) == len(encoder_filter_sizes) == len(encoder_strides)
        assert len(decoder_filters) == len(decoder_filter_sizes) == len(decoder_strides)

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
        self.disable_kl_loss = disable_kl_loss

        self.input_pl = None
        self.input_flat_t = None
        self.mu_t = None
        self.log_var_t = None
        self.sd_t = None
        self.var_t = None
        self.mean_sq_t = None
        self.kl_divergence_t = None
        self.kl_loss_t = None
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
        self.session = None
        self.saver = None

    def encode(self, inputs, batch_size):

        num_steps = int(np.ceil(inputs.shape[0] / batch_size))
        encodings = []

        for step_idx in range(num_steps):

            batch_slice = np.index_exp[step_idx * batch_size:(step_idx + 1) * batch_size]

            tmp_encoding = self.session.run(self.sample_t, feed_dict={
                self.input_pl: inputs[batch_slice]
            })

            encodings.append(tmp_encoding)

        encodings = np.concatenate(encodings, axis=0)

        return encodings

    def predict(self, num_samples):

        outputs = self.session.run(self.output_t, feed_dict={
            self.mu_t: np.zeros((num_samples, self.latent_space_size), dtype=np.float32),
            self.sd_t: np.ones((num_samples, self.latent_space_size), dtype=np.float32)

        })

        return outputs[:, :, :, 0]

    def train(self, samples):

        _, loss, output_loss, kl_loss, reg_loss = self.session.run(
            [self.step_op, self.loss_t, self.output_loss_t, self.kl_loss_t, self.reg_loss_t], feed_dict={
                self.input_pl: samples
            }
        )

        return loss, output_loss, kl_loss, reg_loss

    def build_all(self):

        self.build_placeholders()
        self.build_network()
        self.build_training()

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MODEL_NAMESPACE)
        )

    def build_placeholders(self):

        self.input_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="input_pl")
        self.input_flat_t = tf.reshape(self.input_pl, shape=(tf.shape(self.input_pl)[0], self.flat_input_shape))

    def build_network(self):

        with tf.variable_scope(self.MODEL_NAMESPACE):

            # encoder
            x = self.build_encoder(self.input_pl)

            # middle
            self.sample_t, self.kl_divergence_t, self.mu_t, self.sd_t = self.build_middle(x)

            # decoder
            self.logits_t, self.flat_logits_t, self.output_t = self.build_decoder(self.sample_t)

    def build_encoder(self, input_t, share_weights=False):

        x = tf.expand_dims(input_t, axis=-1)

        with tf.variable_scope("encoder", reuse=share_weights):

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

        return x

    def build_middle(self, input_t, share_weights=False):

        with tf.variable_scope("middle", reuse=share_weights):

            mu_t = tf.layers.dense(
                input_t, self.latent_space_size, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
            )
            log_var_t = tf.layers.dense(
                input_t, self.latent_space_size, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay)
            )

            var_t = tf.exp(log_var_t)
            sd_t = tf.sqrt(var_t)
            mean_sq_t = tf.square(mu_t)

            kl_divergence_t = 0.5 * (mean_sq_t + var_t - log_var_t - 1.0)

            noise_t = tf.random_normal(
                shape=(tf.shape(mu_t)[0], self.latent_space_size), mean=0, stddev=1.0
            )

            sd_noise_t = noise_t * sd_t
            sample_t = mu_t + sd_noise_t

        return sample_t, kl_divergence_t, mu_t, sd_t

    def build_decoder(self, input_t, share_weights=False):

        with tf.variable_scope("decoder", reuse=share_weights):

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

            self.kl_loss_t = tf.reduce_mean(tf.reduce_sum(self.kl_divergence_t, axis=1))

            self.reg_loss_t = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            if self.disable_kl_loss:
                self.loss_t = self.output_loss_t + self.reg_loss_t
            else:
                self.loss_t = self.output_loss_t + self.kl_loss_t + self.reg_loss_t

            self.step_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_t)
