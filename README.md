# Variational Autoencoders in Tensorflow

My implementation is based on [this tutorial](https://arxiv.org/abs/1606.05908) and 
[this open-source caffe code](https://github.com/cdoersch/vae_tutorial).

```
# fully-connected autoencoder on MNIST
python -m vae.scripts.vae_fc_mnist

# convolutional autoencoder on MNIST
python -m vae.scripts.vae_conv_mnist

# categorical convolutional autoencoder with Gumbel-Softmax on MNIST
python -m vae.scripts.sg_vae_conv_mnist
```

Samples from a fully-connected autoencoder:

![samples_fc](vae/results/vae_fc_samples.png)

Samples from a convolutional autoencoder:

![samples_conv](vae/results/vae_conv_samples.png)

Samples from a Gumbel-Softmax Categorical VAE:

Categorical KL divergence:

![cat_gs_vae](vae/results/gs_vae_samples_categorical_kl_500_1e4_samples.png)

Relative KL divergence:

![rel_gs_vae](vae/results/gs_vae_samples_relaxed_kl_500_1e4_samples.png)

Straight-through:

![st_gs_vae](vae/results/gs_vae_samples_straight_through_500_1e4_samples.png)

