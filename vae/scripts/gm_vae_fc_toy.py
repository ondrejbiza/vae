import argparse
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from .. import gm_vae_fc


def main(args):

    # gpu settings
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # generate and show dataset
    sd = 0.1
    num_points = 1000

    train_data = np.concatenate([
        np.random.multivariate_normal([1, 1], [[sd, 0], [0, sd]], size=num_points),
        np.random.multivariate_normal([1, -1], [[sd, 0], [0, sd]], size=num_points),
        np.random.multivariate_normal([-1, 1], [[sd, 0], [0, sd]], size=num_points),
        np.random.multivariate_normal([-1, -1], [[sd, 0], [0, sd]], size=num_points)
    ], axis=0)

    print("training data:")
    plt.scatter(train_data[:, 0], train_data[:, 1])
    plt.show()

    # build model
    num_clusters = 4
    x_size = 2
    w_size = 2
    input_size = 2

    model = gm_vae_fc.GM_VAE(
        input_size, [120, 120], [120, 120, input_size], [120], num_clusters, x_size, w_size,
        gm_vae_fc.GM_VAE.LossType.L2, args.weight_decay, args.learning_rate, clip_z_prior=args.clip_z_prior
    )

    model.build_all()
    model.start_session(gpu_memory=args.gpu_memory_fraction)

    # train model
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

    x_encodings, w_encodings = model.encode(train_data, args.batch_size)
    y_decodings = model.predict_from_x_sample(x_encodings)

    # plot x
    print("encodings to x:")
    plt.scatter(x_encodings[:, 0], x_encodings[:, 1])
    plt.show()

    # plot w
    print("encodings to w:")
    plt.scatter(w_encodings[:, 0], w_encodings[:, 1])
    plt.show()

    # plot reconstruction
    print("reconstructions:")
    plt.scatter(y_decodings[:, 0], y_decodings[:, 1])
    plt.show()

    # plot samples from mixtures
    w_samples = np.random.normal(0, 1, size=(100, w_size))
    c_mu, c_sd = model.get_clusters(w_samples)

    print("cluster centroids:")
    for c_idx in range(num_clusters):
        plt.scatter(c_mu[:, c_idx, 0], c_mu[:, c_idx, 1], label="cluster {:d}".format(c_idx + 1))

    plt.legend()
    plt.show()

    print("cluster samples:")
    for c_idx in range(num_clusters):

        x_samples = c_mu[:, c_idx, :] + np.random.normal(0, 1, size=(100, x_size)) * c_sd[:, c_idx, :]
        y_samples = model.predict_from_x_sample(x_samples)

        plt.scatter(y_samples[:, 0], y_samples[:, 1], label="cluster {:d}".format(c_idx + 1))

    plt.legend()
    plt.show()

    model.stop_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-training-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-z-prior", type=float, default=1.4)

    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-memory-fraction", default=None, type=float)

    parsed = parser.parse_args()
    main(parsed)
