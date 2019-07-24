# Variational Autoencoders in Tensorflow

<p float="left">
    <img src="vae/results/vae_conv_samples.png" alt="vae_mnist_samples" width="420"/>
    <img src="vae/results/vae_conv_losses.png" alt="vae_mnist_losses" width="420"/>
</p>

## Set up

* Install Python >= 3.6.
* Install packages in *requirements.txt*.
* Tested with tensorflow-gpu 1.7.0 (CUDA 9.1, cuDNN 7.1) and tensorflow-gpu 1.14.0 (CUDA 10.0, cuDNN 7.6).
* For tensorflow-gpu 1.14.0, use the flag --fix-cudnn if you get a cuDNN initialization error.
## Usage

### Autoencoder:
```
# ConvNet on MNIST
python -m vae.scripts.ae_conv_mnist
```

MNIST, default settings: -54.26 test log-likelihood (1 run)

### Variational Autoencoder (VAE):

```
# ConvNet on MNIST
python -m vae.scripts.vae_conv_mnist

# fully-connected net on MNIST
python -m vae.scripts.vae_fc_mnist
```

Paper: https://arxiv.org/abs/1312.6114

MNIST, ConvNet, default settings: -71.52 test log-likelihood (1 run)

### VampPrior VAE:

```
# ConvNet on MNIST
python -m vae.scripts.vampprior_vae_conv_mnist

# fully-connected net on a toy dataset
python -m vae.scripts.vampprior_vae_fc_toy
```

Paper: https://arxiv.org/abs/1705.07120

MNIST, default settings: -70.08 test log-likelihood (1 run)

### Gaussian Mixture Prior VAE:

```
# ConvNet on MNIST
python -m vae.scripts.gmprior_vae_conv_mnist

# fully-connected net on a toy dataset
python -m vae.scripts.gmprior_vae_fc_toy
```

Baseline from https://arxiv.org/abs/1705.07120

MNIST, ConvNet, default settings: -69.58 test log-likelihood (1 run)

### Softmax-Gumbel VAE:

```
# ConvNet on MNIST
python -m vae.scripts.sg_vae_conv_mnist
```

Paper: https://arxiv.org/abs/1611.01144

MNIST, default settings: -81.56 test log-likelihood (1 run)

## Notes

* The architecture of all ConvNets is based on this paper (https://arxiv.org/abs/1803.10122) with half the filters.
