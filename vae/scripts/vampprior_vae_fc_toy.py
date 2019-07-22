import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from .. import toy_dataset
from .. import vampprior_vae_fc


def main(args):

    train_data = toy_dataset.get_dataset()
    train_data = shuffle(train_data)

    plt.scatter(train_data[:, 0], train_data[:, 1])
    plt.show()

    model = vampprior_vae_fc.VAMPPRIOR_VAE(
        [2], [16, 16], [16, 16, 2], 2, vampprior_vae_fc.VAMPPRIOR_VAE.LossType.L2, args.learning_rate,
        args.weight_decay, args.num_pseudo_inputs, beta1=1.0, beta2=1.0
    )

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
            losses["entropy loss"].append(np.mean(epoch_losses["entropy loss"]))
            losses["prior loss"].append(np.mean(epoch_losses["prior loss"]))
            losses["regularization"].append(np.mean(epoch_losses["regularization"]))

            epoch_losses = collections.defaultdict(list)

            train_data = shuffle(train_data)

        samples = train_data[epoch_step * batch_size : (epoch_step + 1) * batch_size]

        loss, output_loss, entropy_loss, prior_loss, reg_loss = model.train(samples)

        epoch_losses["total"].append(loss)
        epoch_losses["output"].append(output_loss)
        epoch_losses["entropy loss"].append(entropy_loss)
        epoch_losses["prior loss"].append(prior_loss)
        epoch_losses["regularization"].append(reg_loss)

    samples, pseudo_inputs = model.predict(400)
    model.stop_session()

    # plot pseudo inputs
    plt.scatter(pseudo_inputs[:, 0], pseudo_inputs[:, 1])
    plt.show()

    # plot samples
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()

    # plot losses
    for key, value in losses.items():
        plt.plot(list(range(1, len(value) + 1)), value, label=key)

    plt.legend()
    plt.ylim([-4, 6])
    plt.xlabel("epoch")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-training-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-pseudo-inputs", type=int, default=4)

    parsed = parser.parse_args()
    main(parsed)
