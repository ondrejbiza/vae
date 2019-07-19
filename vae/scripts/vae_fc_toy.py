import collections
import numpy as np
import matplotlib.pyplot as plt
from .. import toy_dataset
from .. import vae_fc


train_data = toy_dataset.get_dataset()

model = vae_fc.VAE([2], [120, 120], [120, 120, 2], 120, vae_fc.VAE.LossType.L2, 0.0005, 0.0001)

model.start_session()

batch_size = 100
epoch_size = len(train_data) // batch_size

losses = collections.defaultdict(list)
epoch_losses = collections.defaultdict(list)

for train_step in range(100000):

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

samples = model.predict(400)
model.stop_session()

# plot samples
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()

# plot losses
for key, value in losses.items():
    plt.plot(list(range(1, len(value) + 1)), value, label=key)

plt.legend()
plt.xlabel("epoch")
plt.show()
