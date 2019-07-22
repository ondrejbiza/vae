import argparse
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from ..vampprior_vae_conv import VAMPPRIOR_VAE
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
    if args.no_pseudo_input_activation:
        pia = None
    else:
        pia = utils.hardtanh

    model = VAMPPRIOR_VAE(
        [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4], [2, 2, 2, 1],
        32, VAMPPRIOR_VAE.LossType.SIGMOID_CROSS_ENTROPY, args.weight_decay, args.learning_rate,
        args.num_pseudo_inputs, pseudo_inputs_activation=pia, beta1=1.0, beta2=1.0, fix_cudnn=args.fix_cudnn
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

        samples = train_data[epoch_step * batch_size : (epoch_step + 1) * batch_size]

        loss, output_loss, entropy_loss, prior_loss, reg_loss = model.train(samples)

        epoch_losses["total"].append(loss)
        epoch_losses["output"].append(output_loss)
        epoch_losses["entropy loss"].append(entropy_loss)
        epoch_losses["prior loss"].append(prior_loss)
        epoch_losses["regularization"].append(reg_loss)

    samples, _ = model.predict(25)
    samples = samples[:, :, :, 0]
    pseudo_inputs = model.get_pseudo_inputs()
    test_lls = model.get_log_likelihood(eval_data)
    model.stop_session()

    print("test negative log-likelihood: {:.2f}".format(np.mean(test_lls)))

    # plot pseudo-inputs
    _, axes = plt.subplots(nrows=5, ncols=5)

    for i in range(25):

        if i >= len(pseudo_inputs):
            break

        axis = axes[i // 5, i % 5]

        axis.imshow(pseudo_inputs[i], vmin=0, vmax=1, cmap="gray")
        axis.axis("off")

    plt.show()

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

    parser.add_argument("--num-training-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-pseudo-inputs", type=int, default=10)
    parser.add_argument("--no-pseudo-input-activation", default=False, action="store_true")

    parser.add_argument("--fix-cudnn", default=False, action="store_true")
    parser.add_argument("--gpus")

    parsed = parser.parse_args()
    main(parsed)
