import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from .. import sg_vae_conv


def main(args):

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    train_data = train_data / 255.0
    eval_data = eval_data / 255.0

    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    # the same settings as in https://arxiv.org/abs/1803.10122, only half the filters
    # in all fully-connected and convolutional layers

    if args.kl_type == "categorical":
        kl_type = sg_vae_conv.SG_VAE.KLType.CATEGORICAL
    else:
        kl_type = sg_vae_conv.SG_VAE.KLType.RELAXED

    def get_temp(step):
        step = (step // args.temp_step) * args.temp_step
        return np.exp(args.temp_mult * step)

    model = sg_vae_conv.SG_VAE(
        [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4], [2, 2, 2, 1],
        20, 10, sg_vae_conv.SG_VAE.LossType.SIGMOID_CROSS_ENTROPY, args.weight_decay, args.learning_rate,
        kl_type, disable_kl_loss=args.disable_kl_loss, straight_through=args.straight_through,
        fix_cudnn=args.fix_cudnn
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
            losses["KL divergence"].append(np.mean(epoch_losses["KL divergence"]))
            losses["regularization"].append(np.mean(epoch_losses["regularization"]))

            epoch_losses = collections.defaultdict(list)

        samples = train_data[epoch_step * batch_size : (epoch_step + 1) * batch_size]

        loss, output_loss, kl_loss, reg_loss = model.train(
            samples, max(args.temp_threshold, get_temp(train_step))
        )

        epoch_losses["total"].append(loss)
        epoch_losses["output"].append(output_loss)
        epoch_losses["KL divergence"].append(kl_loss)
        epoch_losses["regularization"].append(reg_loss)

    samples = model.predict(25)
    test_lls = model.get_log_likelihood(eval_data)
    model.stop_session()

    print("test negative log-likelihood: {:.2f}".format(np.mean(test_lls)))

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--disable-kl-loss", default=False, action="store_true")
    parser.add_argument("--num-training-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--straight-through", default=False, action="store_true")
    parser.add_argument("--kl-type", default="categorical", help="categorical or relaxed")
    parser.add_argument("--temp-mult", type=float, default=-1e-4)
    parser.add_argument("--temp-threshold", type=float, default=0.5)
    parser.add_argument("--temp-step", type=int, default=500)

    parser.add_argument("--fix-cudnn", default=False, action="store_true")

    parsed = parser.parse_args()
    main(parsed)
