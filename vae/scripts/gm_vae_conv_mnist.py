import argparse
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from .. import gm_vae_conv


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
    model = gm_vae_conv.GM_VAE(
        [28, 28], [16, 32, 64], [6, 6, 4], [1, 1, 2], [500], [500], [64, 32, 16, 1], [4, 6, 6, 1], [2, 2, 2, 1],
        [500], 10, 200, 150, gm_vae_conv.GM_VAE.LossType.SIGMOID_CROSS_ENTROPY, args.weight_decay, args.learning_rate,
        clip_z_prior=args.clip_z_prior
    )

    model.build_all()
    model.start_session(gpu_memory=args.gpu_memory_fraction)

    epoch_size = len(train_data) // args.batch_size

    losses = collections.defaultdict(list)
    epoch_losses = collections.defaultdict(list)

    for train_step in range(args.num_training_steps):

        epoch_step = train_step % epoch_size

        if train_step > 0 and train_step % 1000 == 0:
            print("step {:d}".format(train_step))

        if epoch_step == 0 and train_step > 0:

            losses["total"].append(np.mean(epoch_losses["total"]))
            losses["output"].append(np.mean(epoch_losses["output"]))
            losses["w KL divergence"].append(np.mean(epoch_losses["w KL divergence"]))
            losses["x KL divergence"].append(np.mean(epoch_losses["x KL divergence"]))
            losses["z KL divergence"].append(np.mean(epoch_losses["z KL divergence"]))
            losses["regularization"].append(np.mean(epoch_losses["regularization"]))

            epoch_losses = collections.defaultdict(list)

        samples = train_data[epoch_step * args.batch_size : (epoch_step + 1) * args.batch_size]

        loss, output_loss, w_kl_loss, x_kl_loss, z_kl_loss, reg_loss = model.train(samples)

        epoch_losses["total"].append(loss)
        epoch_losses["output"].append(output_loss)
        epoch_losses["w KL divergence"].append(w_kl_loss)
        epoch_losses["x KL divergence"].append(x_kl_loss)
        epoch_losses["z KL divergence"].append(z_kl_loss)
        epoch_losses["regularization"].append(reg_loss)

    samples = model.predict(25)

    # plot samples
    _, axes = plt.subplots(nrows=5, ncols=5)

    for i in range(25):

        axis = axes[i // 5, i % 5]

        axis.imshow(samples[i], vmin=0, vmax=1, cmap="gray")
        axis.axis("off")

    plt.show()

    # plot losses
    for key, value in losses.items():
        plt.plot(list(range(1, len(value) + 1)), value, label=key)

    plt.legend()
    plt.xlabel("epoch")
    plt.show()

    # plot samples from mixtures
    _, axes = plt.subplots(nrows=10, ncols=10)

    c_mu, c_sd = model.get_clusters()

    for c_idx in range(10):
        for s_idx in range(10):

            idx = c_idx * 10 + s_idx

            x_sample = c_mu[c_idx] + np.random.normal(0, 1, size=200) * c_sd[c_idx]
            y_sample = model.predict_from_x_sample(x_sample[np.newaxis, :])[0]

            axis = axes[idx // 10, idx % 10]

            axis.imshow(y_sample, vmin=0, vmax=1, cmap="gray")
            axis.axis("off")

    plt.show()

    model.stop_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-training-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-z-prior", type=float, default=None)

    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-memory-fraction", default=None, type=float)

    parsed = parser.parse_args()
    main(parsed)
