import os
from enum import Enum
import numpy as np
import tensorflow as tf
from . import utils


class GM_VAE:

    MODEL_NAMESPACE = "model"
    TRAINING_NAMESPACE = "training"

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    def __init__(self, input_shape, encoder_filters, encoder_filter_sizes, encoder_strides, encoder_neurons,
                 decoder_neurons, decoder_filters, decoder_filter_sizes, decoder_strides, cluster_predictor_neurons,
                 num_clusters, x_size, w_size, loss_type, weight_decay, learning_rate,
                 cluster_predictor_activation=tf.nn.tanh, clip_z_prior=None, use_bn=True):

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
        self.cluster_predictor_neurons = cluster_predictor_neurons
        self.num_clusters = num_clusters
        self.x_size = x_size
        self.w_size = w_size
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.cluster_predictor_activation = cluster_predictor_activation
        self.clip_z_prior = clip_z_prior
        self.use_bn = use_bn

        self.input_pl = None
        self.input_flat_t = None
        self.w_kl_divergence_t = None
        self.w_kl_loss_t = None
        self.x_kl_divergence_t = None
        self.x_kl_loss_t = None
        self.z_kl_divergence_t = None
        self.z_kl_loss_t = None
        self.reg_loss_t = None
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

            tmp_encoding = self.session.run(self.x_sample_t, feed_dict={
                self.input_pl: inputs[batch_slice],
                self.is_training_pl: False
            })

            encodings.append(tmp_encoding)

        encodings = np.concatenate(encodings, axis=0)

        return encodings

    def predict(self, num_samples):

        outputs = self.session.run(self.output_t, feed_dict={
            self.x_mu_t: np.zeros((num_samples, self.x_size), dtype=np.float32),
            self.x_sd_t: np.ones((num_samples, self.x_size), dtype=np.float32),
            self.is_training_pl: False
        })

        return outputs[:, :, :, 0]

    def predict_from_x_sample(self, x_samples):

        outputs = self.session.run(self.output_t, feed_dict={
            self.x_sample_t: x_samples,
            self.is_training_pl: False
        })

        return outputs[:, :, :, 0]

    def get_clusters(self):

        c_mu, c_sd = self.session.run([self.c_mu_t, self.c_sd_t], feed_dict={
            self.w_sample_t: np.zeros((1, self.w_size), dtype=np.float32),
            self.is_training_pl: False
        })

        return c_mu[0], c_sd[0]

    def train(self, samples):

        _, loss, output_loss, w_kl_loss, x_kl_loss, z_kl_loss, reg_loss = self.session.run(
            [self.step_op, self.loss_t, self.output_loss_t, self.w_kl_loss_t, self.x_kl_loss_t, self.z_kl_loss_t,
             self.reg_loss_t], feed_dict={
                self.input_pl: samples,
                self.is_training_pl: True
            }
        )

        return loss, output_loss, w_kl_loss, x_kl_loss, z_kl_loss, reg_loss

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
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_network(self):

        with tf.variable_scope(self.MODEL_NAMESPACE):

            # encoder
            x = self.build_encoder(self.input_pl)

            # middle
            self.x_sample_t, self.x_mu_t, self.x_sd_t, self.x_var_t, w_mu_t, w_sd_t, self.w_sample_t, self.c_mu_t, \
                self.c_sd_t, self.w_kl_divergence_t, self.x_kl_divergence_t, self.z_kl_divergence_t = \
                self.build_middle(x)

            # decoder
            self.logits_t, self.flat_logits_t, self.output_t = self.build_decoder(self.x_sample_t)

    def build_encoder(self, input_t, share_weights=False):

        x = tf.expand_dims(input_t, axis=-1)

        with tf.variable_scope("encoder", reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.use_bn else None,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                    )

                    if self.use_bn:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.use_bn else None,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                    )

                    if self.use_bn:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

        return x

    def build_cluster_predictor(self, input_t, shape_weights=False):

        x = input_t

        with tf.variable_scope("cluster_predictor", reuse=shape_weights):

            for idx, neurons in enumerate(self.cluster_predictor_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=self.cluster_predictor_activation,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                    )

        return x

    def build_middle(self, input_1_t, share_weights=False):

        with tf.variable_scope("middle", reuse=share_weights):

            # predict x
            x_mu_t = tf.layers.dense(
                input_1_t, self.x_size, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            x_log_var_t = tf.layers.dense(
                input_1_t, self.x_size, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            x_var_t = tf.exp(x_log_var_t)
            x_sd_t = tf.sqrt(x_var_t)

            # sample x
            x_noise_t = tf.random_normal(
                shape=(tf.shape(x_mu_t)[0], self.x_size), mean=0, stddev=1.0
            )
            x_sample_t = x_mu_t + x_noise_t * x_sd_t

            # predict w
            w_mu_t = tf.layers.dense(
                input_1_t, self.w_size, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            w_log_var_t = tf.layers.dense(
                input_1_t, self.w_size, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            w_var_t = tf.exp(w_log_var_t)
            w_sd_t = tf.sqrt(w_var_t)

            # sample w
            w_noise_t = tf.random_normal(
                shape=(tf.shape(w_mu_t)[0], self.w_size), mean=0, stddev=1.0
            )
            w_sample_t = w_mu_t + w_noise_t * w_sd_t

            # predict clusters
            input_2_t = self.build_cluster_predictor(w_sample_t)

            c_mu_t = tf.layers.dense(
                input_2_t, self.x_size * self.num_clusters, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            c_mu_t = tf.reshape(c_mu_t, shape=(tf.shape(c_mu_t)[0], self.num_clusters, self.x_size))
            c_log_var_t = tf.layers.dense(
                input_2_t, self.x_size * self.num_clusters, activation=None,
                kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            c_log_var_t = tf.reshape(c_log_var_t, shape=(tf.shape(c_log_var_t)[0], self.num_clusters, self.x_size))
            c_var_t = tf.exp(c_log_var_t)
            c_sd_t = tf.square(c_var_t)

            # predict z from x and clusters
            #z_pred_exp = (1 / 2) * tf.reduce_sum(
            #    tf.pow(c_mu_t - x_sample_t[:, tf.newaxis, :], 2) / c_var_t, axis=2
            #)
            #z_pred_exp -= tf.reduce_max(z_pred_exp, axis=1)[:, tf.newaxis]
            #z_pred_prob = tf.reduce_prod(1 / c_var_t, axis=2) * tf.exp(- z_pred_exp)
            #z_pred_prob /= tf.reduce_sum(z_pred_prob, axis=1)[:, tf.newaxis]

            z_pred_logits = - (self.num_clusters / 2) * np.log(2 * np.pi) - \
                (1 / 2) * tf.reduce_sum(c_log_var_t, axis=2) - \
                (1 / 2) * tf.reduce_sum(tf.pow(c_mu_t - x_sample_t[:, tf.newaxis, :], 2) / c_var_t, axis=2)
            z_pred_softmax = tf.nn.softmax(z_pred_logits, axis=1)
            z_pred_logsoftmax = tf.nn.log_softmax(z_pred_logits, axis=1)

            # kl divergences
            w_kl_divergence_t = 0.5 * (tf.square(w_mu_t) + w_var_t - w_log_var_t - 1.0)

            x_kl_divergence_t = 0.5 * (
                c_log_var_t - x_log_var_t[:, tf.newaxis, :] - 1.0 + (x_var_t[:, tf.newaxis, :] / c_var_t) +
                tf.pow(x_mu_t[:, tf.newaxis, :] - c_mu_t, 2) / c_var_t
            ) * z_pred_softmax[:, :, tf.newaxis]

            z_kl_divergence_t = z_pred_softmax * (z_pred_logsoftmax + np.log(self.num_clusters))

        return x_sample_t, x_mu_t, x_sd_t, x_var_t, w_mu_t, w_sd_t, w_sample_t, c_mu_t, c_sd_t, w_kl_divergence_t, \
            x_kl_divergence_t, z_kl_divergence_t

    def build_decoder(self, input_t, share_weights=False):

        with tf.variable_scope("decoder", reuse=share_weights):

            x = input_t

            for idx, neurons in enumerate(self.decoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.use_bn else None,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                    )

                    if self.use_bn:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = x[:, tf.newaxis, tf.newaxis, :]

            for idx in range(len(self.decoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):

                    last = idx == len(self.decoder_filters) - 1

                    x = tf.layers.conv2d_transpose(
                        x, self.decoder_filters[idx], self.decoder_filter_sizes[idx], self.decoder_strides[idx],
                        padding="VALID", activation=tf.nn.relu if not (last or self.use_bn) else None,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.contrib.layers.xavier_initializer()
                    )

                    if self.use_bn and not last:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

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
                self.output_loss_t = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.losses.sigmoid_cross_entropy(
                            multi_class_labels=self.input_flat_t, logits=self.flat_logits_t,
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
                            labels=self.input_flat_t, predictions=self.flat_logits_t,
                            reduction=tf.losses.Reduction.NONE
                        ),
                        axis=1
                    ),
                    axis=0
                )

            self.w_kl_loss_t = tf.reduce_mean(tf.reduce_sum(self.w_kl_divergence_t, axis=1), axis=0)

            self.x_kl_loss_t = tf.reduce_mean(tf.reduce_sum(self.x_kl_divergence_t, axis=[1, 2]), axis=0)

            self.z_kl_loss_t = tf.reduce_mean(tf.reduce_sum(self.z_kl_divergence_t, axis=1), axis=0)

            if self.clip_z_prior is not None:
                self.z_kl_loss_t = tf.reduce_max([self.z_kl_loss_t, self.clip_z_prior])

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if len(reg_losses) == 0:
                self.reg_loss_t = tf.constant(0.0, dtype=tf.float32)
            else:
                self.reg_loss_t = tf.add_n(reg_losses)

            self.loss_t = self.output_loss_t + self.w_kl_loss_t + self.x_kl_loss_t + self.z_kl_loss_t + self.reg_loss_t

            self.step_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_t)

            if self.use_bn:
                self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                self.step_op = tf.group(self.step_op, self.update_op)

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()

    def save(self, path):

        dir_name = os.path.dirname(path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        self.saver.save(self.session, path)

    def load(self, path):

        self.saver.restore(self.session, path)
