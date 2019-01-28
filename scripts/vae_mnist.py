import collections
import matplotlib.pyplot as plt
import tensorflow as tf
from vae import VAE

((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
train_data = train_data / 255.0
eval_data = eval_data / 255.0

model = VAE([28, 28], [1000, 500, 250], [250, 500, 28 * 28], 30, VAE.LossType.SIGMOID_CROSS_ENTROPY, 0.0005, 0.001)

model.start_session()

batch_size = 100
epoch_size = len(train_data) // batch_size

losses = collections.defaultdict(list)

for train_step in range(60000):

    epoch_step = train_step % epoch_size

    if train_step > 0 and train_step % 1000 == 0:
        print("step {:d}".format(train_step))

    samples = train_data[epoch_step * batch_size : (epoch_step + 1) * batch_size]

    loss, output_loss, kl_loss = model.train(samples)

    losses["total"].append(loss)
    losses["output"].append(output_loss)
    losses["KL divergence"].append(kl_loss)

samples = model.predict(25)
model.stop_session()

# plot samples
_, axes = plt.subplots(nrows=5, ncols=5)

for i in range(25):

    axis = axes[i // 5, i % 5]

    axis.imshow(samples[i], vmin=0, vmax=1)
    axis.axis("off")

plt.show()

# plot losses
for key, value in losses.items():
    plt.plot(value, label=key)

plt.legend()
plt.show()
