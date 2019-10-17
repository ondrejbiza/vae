import argparse
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from ..potential_prior_vae_conv import POTENTIAL_PRIOR_VAE


def main(args):

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    train_data = train_data / 255.0
    eval_data = eval_data / 255.0

    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    model = POTENTIAL_PRIOR_VAE(
        [28, 28], [16, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [], [512], [64, 32, 16, 1], [4, 5, 5, 4], [2, 2, 2, 1],
        args.latent_space_size, POTENTIAL_PRIOR_VAE.LossType.SIGMOID_CROSS_ENTROPY, args.prior_type, args.weight_decay,
        args.learning_rate, args.num_components, beta1=args.beta1, beta2=args.beta2, tau=args.tau,
        fix_cudnn=args.fix_cudnn
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
            losses["mu norms"].append(np.mean(epoch_losses["mu norms"]))
            losses["mixture mu norms"].append(np.mean(epoch_losses["mixture mu norms"]))

            epoch_losses = collections.defaultdict(list)

        samples = train_data[epoch_step * batch_size: (epoch_step + 1) * batch_size]

        loss, output_loss, entropy_loss, prior_loss, reg_loss, n1, n2 = model.train(samples)

        #print(loss, output_loss, entropy_loss, prior_loss)

        epoch_losses["total"].append(loss)
        epoch_losses["output"].append(output_loss)
        epoch_losses["entropy loss"].append(entropy_loss)
        epoch_losses["prior loss"].append(prior_loss)
        epoch_losses["regularization"].append(reg_loss)
        epoch_losses["mu norms"].append(n1)
        epoch_losses["mixture mu norms"].append(n2)

    test_lls = model.get_log_likelihood(eval_data)

    print("test negative log-likelihood: {:.2f}".format(np.mean(test_lls)))

    # plot samples
    images = train_data[:args.num_components * 5]
    potentials = model.session.run(model.sample_potential_t, feed_dict={
        model.input_pl: images
    })
    assignment = np.argmax(potentials, axis=1)
    _, counts = np.unique(assignment, return_counts=True)
    max_images = np.max(counts)

    print("max images", max_images)

    _, axes = plt.subplots(nrows=args.num_components, ncols=max_images)

    for i in range(args.num_components):
        mask = assignment == i
        print("component {:d}, {:d} elements".format(i + 1, np.sum(mask)))
        if np.any(mask):
            for j, image in enumerate(images[mask]):
                axis = axes[i, j]
                axis.imshow(image, vmin=0, vmax=1, cmap="gray")
                axis.axis("off")

        for j in range(int(np.sum(mask)), max_images):
            axis = axes[i, j]
            axis.axis("off")

    plt.show()

    # plot losses
    for key, value in losses.items():
        plt.plot(list(range(1, len(value) + 1)), value, label=key)

    plt.legend()
    plt.xlabel("epoch")
    plt.show()

    model.stop_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-training-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-components", type=int, default=100)
    parser.add_argument("--latent-space-size", type=int, default=32)
    parser.add_argument("--prior-type", type=int, default=0, help="1: exp tanh, 2: exp cosine sim")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=1.0)

    parser.add_argument("--fix-cudnn", default=False, action="store_true")
    parser.add_argument("--gpus")

    parsed = parser.parse_args()
    main(parsed)
