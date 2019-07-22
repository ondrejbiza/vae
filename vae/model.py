import os
from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto


class Model(ABC):

    def __init__(self):

        self.input_pl = NotImplemented
        self.full_output_loss_t = NotImplemented
        self.session = NotImplemented
        self.saver = NotImplemented

    def get_log_likelihood(self, data, batch_size=100):

        num_steps = int(np.ceil(data.shape[0] / batch_size))
        lls = []

        for idx in range(num_steps):

            ll = self.session.run(self.full_output_loss_t, feed_dict={
                self.input_pl: data[idx * batch_size: (idx + 1) * batch_size]
            })

            lls.append(ll)

        lls = - np.concatenate(lls, axis=0)

        return lls

    def start_session(self):

        # prevents a cuDNN initialization error in TF 1.14 with CUDA 10.0 and cuDNN 7.6
        config = ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)
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
