# https://github.com/nadavbh12/VQ-VAE
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
# https://github.com/hiwonjoon/tf-vqvae/blob/master/model.py
# https://arxiv.org/pdf/1711.00937.pdf
# http://bayesiandeeplearning.org/2017/papers/54.pdf

from enum import Enum
import numpy as np
import tensorflow as tf
from . import utils
from .model import Model


class VQ_VAE(Model):

    MODEL_NAMESPACE = "model"
    TRAINING_NAMESPACE = "training"

    class LossType(Enum):

        SIGMOID_CROSS_ENTROPY = 1
        L2 = 2

    def __init__(self, input_shape, encoder_filters, encoder_filter_sizes, encoder_strides,
                 decoder_filters, decoder_filter_sizes, decoder_strides, num_embeddings, loss_type,
                 weight_decay, learning_rate, beta1, beta2, lr_decay_val=None, lr_decay_steps=None, fix_cudnn=False):

        super(VQ_VAE, self).__init__(fix_cudnn=fix_cudnn)

        assert loss_type in self.LossType
        assert len(encoder_filters) == len(encoder_filter_sizes) == len(encoder_strides)
        assert len(decoder_filters) == len(decoder_filter_sizes) == len(decoder_strides)

        self.input_shape = input_shape
        self.flat_input_shape = int(np.prod(self.input_shape))
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.decoder_filters = decoder_filters
        self.decoder_filter_sizes = decoder_filter_sizes
        self.decoder_strides = decoder_strides
        self.num_embeddings = num_embeddings
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_decay_val = lr_decay_val
        self.lr_decay_steps = lr_decay_steps

        self.input_pl = None
        self.input_flat_t = None
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

            tmp_encoding = self.session.run(self.classes, feed_dict={
                self.input_pl: inputs[batch_slice]
            })

            encodings.append(tmp_encoding)

        encodings = np.concatenate(encodings, axis=0)

        return encodings

    def decode(self, classes):

        outputs = self.session.run(self.output_t, feed_dict={
            self.classes: classes,
            self.pred_embeds: np.zeros([item.value for item in self.pred_embeds.shape])
        })

        return outputs[:, :, :, 0]

    def predict(self, num_samples):

        classes = np.random.randint(0, self.num_embeddings, size=(num_samples, self.pred_embeds.shape[1].value, self.pred_embeds.shape[2].value))

        outputs = self.session.run(self.output_t, feed_dict={
            self.classes: classes,
            self.pred_embeds: np.zeros([item.value for item in self.pred_embeds.shape])
        })

        return outputs[:, :, :, 0], classes

    def train(self, samples):

        _, loss, output_loss, left_loss, reg_loss, e, p = self.session.run(
            [self.step_op, self.loss_t, self.output_loss_t, self.left_loss_t, self.reg_loss_t, self.embeds, self.pred_embeds],
            feed_dict={
                self.input_pl: samples
            }
        )

        """
        p = np.reshape(p, (p.shape[0] * p.shape[1] * p.shape[2], p.shape[3]))
        a = np.concatenate([e, p], axis=0)

        l = np.concatenate([np.zeros(len(e)), np.ones(len(p))], axis=0)

        from sklearn.decomposition import PCA
        m = PCA(n_components=2)
        x = m.fit_transform(a)

        import matplotlib.pyplot as plt
        plt.scatter(x[:, 0], x[:, 1], c=l)
        plt.show()
        """

        return loss, output_loss, left_loss, reg_loss

    def build_all(self):

        self.build_placeholders()
        self.build_network()
        self.build_training()

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MODEL_NAMESPACE)
        )

    def build_placeholders(self):

        self.input_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="input_pl")
        self.input_resized_t = tf.image.resize_images(
            tf.reshape(self.input_pl, [-1, 28, 28, 1]), (24, 24), method=tf.image.ResizeMethod.BILINEAR
        )

    def build_network(self):

        with tf.variable_scope(self.MODEL_NAMESPACE):

            # encoder
            self.pred_embeds = self.build_encoder(self.input_resized_t)

            # middle
            self.build_middle(self.pred_embeds)

            # decoder
            self.logits_t, self.output_t = self.build_decoder(self.collected_embeds)

    def build_encoder(self, input_t, share_weights=False):

        x = input_t
        print("encoder input:", x)

        with tf.variable_scope("encoder", reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
                    )
                    print("encoder x:", x)

        return x

    def build_middle(self, input_t, share_weights=False):

        with tf.variable_scope("middle", reuse=share_weights):

            self.embeds = tf.get_variable(
                "embeddings", [self.num_embeddings, self.pred_embeds.shape[3].value],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )

            self.diff = input_t[:, :, :, tf.newaxis, :] - self.embeds[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
            self.norm = tf.norm(self.diff, axis=4)
            self.classes = tf.argmin(self.norm, axis=3)

            self.collected_embeds = tf.gather(self.embeds, self.classes)

            # fake gradients
            self.collected_embeds_fake_grads = tf.stop_gradient(self.collected_embeds - input_t) + input_t

    def build_decoder(self, input_t, share_weights=False):

        x = input_t
        print("decoder input:", x)

        with tf.variable_scope("decoder", reuse=share_weights):

            for idx in range(len(self.decoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d_transpose(
                        x, self.decoder_filters[idx], self.decoder_filter_sizes[idx], self.decoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if idx != len(self.decoder_filters) - 1 else None,
                        kernel_regularizer=utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=tf.random_normal_initializer(stddev=0.02)
                    )
                    print("decoder x:", x)

        logits_t = tf.nn.sigmoid(x)
        output_t = logits_t

        #if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
        #    output_t = tf.nn.sigmoid(logits_t)
        #else:
        #   output_t = logits_t

        return logits_t, output_t

    def build_training(self):

        with tf.variable_scope(self.TRAINING_NAMESPACE):

            if self.loss_type == self.LossType.SIGMOID_CROSS_ENTROPY:
                self.full_output_loss_t = tf.reduce_mean(
                    tf.losses.sigmoid_cross_entropy(
                        multi_class_labels=self.input_flat_t, logits=self.flat_logits_t,
                        reduction=tf.losses.Reduction.NONE
                    ),
                    axis=1
                )
            else:
                """
                self.full_output_loss_t = tf.reduce_mean(
                    tf.losses.mean_squared_error(
                        labels=self.input_flat_t, predictions=self.flat_logits_t,
                        reduction=tf.losses.Reduction.NONE
                    ),
                    axis=1
                )
                """
                self.full_output_loss_t = tf.reduce_mean(
                    tf.square((self.input_resized_t - self.logits_t)), axis=[1, 2, 3]
                )

            self.output_loss_t = tf.reduce_mean(self.full_output_loss_t, axis=0)

            self.left_loss_t = tf.reduce_mean(
                tf.norm(tf.stop_gradient(self.pred_embeds) - self.collected_embeds, axis=3) ** 2
            )

            self.right_loss_t = tf.reduce_mean(
                tf.norm(self.pred_embeds - tf.stop_gradient(self.collected_embeds), axis=3) ** 2
            )

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if len(reg_losses) > 0:
                self.reg_loss_t = tf.add_n(reg_losses)
            else:
                self.reg_loss_t = tf.constant(0.0, dtype=tf.float32)

            self.loss_t = self.output_loss_t + self.beta1 * self.left_loss_t + self.beta2 * self.right_loss_t + \
                self.reg_loss_t

            self.global_step = tf.train.get_or_create_global_step()

            if self.lr_decay_val is not None and self.lr_decay_steps is not None:
                learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.lr_decay_steps, self.lr_decay_val)
            else:
                learning_rate = self.learning_rate

            self.step_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                self.loss_t, global_step=self.global_step
            )

    def embedding_difference(self, predictions, actual):

        return predictions[:, :, tf.newaxis, :] - actual[tf.newaxis, tf.newaxis, :, :]
