import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from .. import vae_fc

((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
train_data = train_data / 255.0
eval_data = eval_data / 255.0

train_data, train_labels = shuffle(train_data, train_labels)
eval_data, eval_labels = shuffle(eval_data, eval_labels)

model = vae_fc.VAE([28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, vae_fc.VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001)

model.start_session()

batch_size = 100
epoch_size = len(train_data) // batch_size

losses = collections.defaultdict(list)
epoch_losses = collections.defaultdict(list)

for train_step in range(60000):

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

    loss, output_loss, kl_loss, reg_loss = model.train(samples)

    epoch_losses["total"].append(loss)
    epoch_losses["output"].append(output_loss)
    epoch_losses["KL divergence"].append(kl_loss)
    epoch_losses["regularization"].append(reg_loss)

samples = model.predict(25)
model.stop_session()

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
