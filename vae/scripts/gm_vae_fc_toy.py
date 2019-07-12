import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from .. import gm_vae_fc


def main(args):

    sd = 0.1

    train_data = np.concatenate([
        np.random.multivariate_normal([1, 1], [[sd, 0], [0, sd]], size=100),
        np.random.multivariate_normal([1, -1], [[sd, 0], [0, sd]], size=100),
        np.random.multivariate_normal([-1, 1], [[sd, 0], [0, sd]], size=100),
        np.random.multivariate_normal([-1, -1], [[sd, 0], [0, sd]], size=100)
    ], axis=0)

    #plt.scatter(train_data[:, 0], train_data[:, 1])
    #plt.show()

    # the same settings as in https://arxiv.org/abs/1803.10122, only half the filters
    # in all fully-connected and convolutional layers
    model = gm_vae_fc.GM_VAE(
        2, [120, 120], [120, 120, 2], [120], 4, 2, 2, gm_vae_fc.GM_VAE.LossType.L2, args.weight_decay,
        args.learning_rate, clip_z_prior=args.clip_z_prior
    )

    model.build_all()
    model.start_session()

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

    # plot losses
    for key, value in losses.items():
        plt.plot(list(range(1, len(value) + 1)), value, label=key)

    plt.legend()
    plt.xlabel("epoch")
    plt.show()

    # plot x
    print("x samples")
    x_samples = model.encode(train_data, args.batch_size)

    plt.scatter(x_samples[:, 0], x_samples[:, 1])
    plt.show()

    # plot reconstruction
    print("y samples")
    y_samples = model.predict_from_x_sample(x_samples)

    plt.scatter(y_samples[:, 0], y_samples[:, 1])
    plt.show()

    # plot samples from mixtures
    samples = []
    classes = []

    c_mu, c_sd = model.get_clusters()

    print("clusters")
    print(c_mu.shape)
    print(c_sd)
    plt.scatter(c_mu[:, 0], c_mu[:, 1])
    plt.show()

    for c_idx in range(4):
        for s_idx in range(100):

            x_sample = c_mu[c_idx] + np.random.normal(0, 1, size=2) * c_sd[c_idx]
            y_sample = model.predict_from_x_sample(x_sample[np.newaxis, :])[0]

            samples.append(y_sample)
            classes.append(c_idx + 1)

    samples = np.stack(samples, axis=0)
    classes = np.array(classes, dtype=np.int32)

    plt.scatter(samples[:, 0], samples[:, 1], c=classes)
    plt.show()

    model.stop_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--disable-kl-loss", default=False, action="store_true")
    parser.add_argument("--num-training-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-z-prior", type=float, default=None)

    parsed = parser.parse_args()
    main(parsed)
