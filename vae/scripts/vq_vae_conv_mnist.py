import argparse
import collections
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from .. import vq_vae_conv
from .. import utils


def main(args):

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    train_data = train_data / 255.0
    eval_data = eval_data / 255.0

    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    # the same settings as in https://arxiv.org/abs/1803.10122, only half the filters
    # in all fully-connected and convolutional layers
    model = vq_vae_conv.VQ_VAE(
        [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4], [2, 2, 2, 1],
        args.latent_size, args.num_embeddings, args.embedding_size, vq_vae_conv.VQ_VAE.LossType.SIGMOID_CROSS_ENTROPY,
        args.weight_decay, args.learning_rate, args.beta1, args.beta2
    )

    model.build_all()
    model.start_session()

    batch_size = 100
    epoch_size = len(train_data) // batch_size

    losses = collections.defaultdict(list)
    epoch_losses = collections.defaultdict(list)

    for train_step in range(args.num_training_steps):

        epoch_step = train_step % epoch_size

        if train_step > 0 and train_step % 1000 == 0:
            print("step {:d}".format(train_step))

        if epoch_step == 0 and train_step > 0:

            losses["total"].append(np.mean(epoch_losses["total"]))
            losses["output"].append(np.mean(epoch_losses["output"]))
            losses["commitment loss"].append(np.mean(epoch_losses["commitment loss"]))
            losses["regularization"].append(np.mean(epoch_losses["regularization"]))

            epoch_losses = collections.defaultdict(list)

        samples = train_data[epoch_step * batch_size : (epoch_step + 1) * batch_size]

        loss, output_loss, commitment_loss, reg_loss = model.train(samples)

        epoch_losses["total"].append(loss)
        epoch_losses["output"].append(output_loss)
        epoch_losses["commitment loss"].append(commitment_loss)
        epoch_losses["regularization"].append(reg_loss)

    valid_batch = eval_data[0: 25]
    valid_classes = model.encode(valid_batch, 25)
    valid_reconstructions = model.decode(valid_classes)

    print("inputs:")

    utils.plot_square(valid_batch)
    plt.show()

    print("classes:", valid_classes)
    print("reconstructions:")

    utils.plot_square(valid_reconstructions)
    plt.show()

    samples, classes = model.predict(25)
    test_lls = model.get_log_likelihood(eval_data)
    model.stop_session()

    print("test negative log-likelihood: {:.2f}".format(np.mean(test_lls)))

    # plot samples
    print("sample classes:", classes)
    utils.plot_square(samples)
    plt.show()

    # plot losses
    for key, value in losses.items():
        plt.plot(list(range(1, len(value) + 1)), value, label=key)

    plt.legend()
    plt.xlabel("epoch")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--num-embeddings", type=int, default=10)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--disable-kl-loss", default=False, action="store_true")
    parser.add_argument("--num-training-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--beta1", type=float, default=0.25)
    parser.add_argument("--beta2", type=float, default=0.25)

    parser.add_argument("--gpus")

    parsed = parser.parse_args()
    main(parsed)
