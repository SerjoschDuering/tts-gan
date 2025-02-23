# TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network
---

This repository contains code from the paper "TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network".

The paper has been accepted to publish in the 20th International Conference on Artificial Intelligence in Medicine (AIME 2022).

Please find the paper [here](https://arxiv.org/abs/2202.02691)

---
# Hackathon Additions

# High-Level Overview

This tweaked TTS-GAN architecture builds on a transformer‐based GAN design and augments it with a conditional autoencoder (with VAE properties) to anonymize long time series data. The overall idea is to first embed real signals into a latent space using an encoder, cluster these latent representations to capture similar patterns, and then use the averaged, cluster‐derived latent conditions to drive a conditional generator. The generator ultimately synthesizes realistic, anonymized signals that capture the key temporal dynamics of the original data without revealing individual identities.

![Screenshot of the TTS-GAN architecture](Screenshot%202025-02-23%20at%209.26.50%20AM.jpg)

---

## Introduction

The goal is to anonymize long time-series data (originally of dimension 8760×1, later reduced to 2190×1) using a conditional generative model. By combining a VAE‐like encoder, which learns latent embeddings with sampling capabilities, and a transformer-based GAN, the system learns to generate synthetic signals conditioned on cluster‐averaged characteristics. This ensures that while the generated data preserves statistical and temporal properties of the original signals, it removes sensitive identity-specific details.

---

## Workflow Illustration

1. **Encoding and Clustering**
   • Each original time series is fed into the Encoder module that uses patch embeddings and a transformer encoder.
   • The encoder computes mean and log-variance parameters and then samples a latent representation via the reparameterization trick (VAE style).
   • These latent embeddings are clustered, grouping signals with similar temporal properties.

2. **Condition Formation**
   • For every identified cluster, several latent probes are sampled and averaged.
   • This averaged latent vector acts as the condition input that captures the characteristic pattern of the cluster.

3. **Conditional Generation**
   • The Generator receives both a random noise vector and the condition vector (obtained from clustering).
   • Through a series of linear projections, positional embeddings, and multi-scale transformer encoder blocks (full, medium, and small branches), the generator crafts a detailed time-series representation.
   • A final deconvolution layer (with spectral normalization and Tanh activation) processes the combined features into the synthetic (anonymized) time series data.

4. **Adversarial Training**
   • A discriminator (also built with transformer layers and patch embeddings) contrasts real versus synthetic signals.
   • The adversarial losses (including options for gradient penalty and various loss functions) drive the training loop, ensuring the generator produces realistic sequences.

---

## Architecture Features (Related to the Task)

- **Transformer-Based Modules:**
  Both generator and discriminator leverage transformer encoder blocks that capture long-term temporal dependencies. The multi-head self-attention mechanism enables modeling of complex time series relationships over long sequences.

- **Multi-Scale Processing:**
  The generator incorporates a multi-scale design by processing the time series at full, medium, and small scales. Adaptive pooling and separate positional embeddings for each resolution help refine the synthesis of fine-grained details and overall structure.

- **Conditional Generation with VAE Properties:**
  A dedicated Encoder module produces latent representations with a VAE sampling mechanism (mean, log variance, reparameterization). These latent variables can be clustered and averaged to form condition vectors, guiding the generator to produce anonymized outputs while preserving structural similarity to the original data.

- **Stabilization Techniques:**
  Techniques like spectral normalization on linear and convolutional layers, residual connections, and gradient penalties in the discriminator loss contribute to training stability, which is important in GAN setups especially for long sequence synthesis.

- **Flexible Configuration:**
  The code offers a rich set of parameters (configurable via the `cfg.py` file and command-line arguments in `RunningGAN_Train.py`), from sequence length and patch sizes to learning rates and transformer depths—allowing adaptation to different data modalities and resolutions.

---

## Shortcomings

- **Interface Complexity:**
  The Generator’s forward method strictly requires separate noise and conditional inputs. In some parts of the training code, only a noise vector is passed, which may lead to interface mismatches if not carefully managed.

- **Computational Overhead:**
  Multi-scale transformer architectures processing long time series can be computationally expensive and memory-intensive, which may limit scalability without further optimization or hardware support (e.g., mixed precision training using the FP16 flag).

- **Sensitivity to Hyperparameters:**
  The performance of transformers and GANs is known to be sensitive to hyperparameter choices. Adapting parameters such as dropout rates, latent dimensions, and clustering thresholds requires careful tuning to avoid issues like mode collapse or overfitting.

---

## Risks

- **Anonymization Effectiveness:**
  The final anonymization relies on the quality of the latent space clustering. Poor clustering may lead to conditions that do not fully disconnect individual-specific features, potentially risking privacy.

- **Training Stability:**
  GANs are notoriously fickle, and the addition of t



---

**Abstract:**
Time-series datasets used in machine learning applications often are small in size, making the training of deep neural network architectures ineffective. For time series, the suite of data augmentation tricks we can use to expand the size of the dataset is limited by the need to maintain the basic properties of the signal. Data generated by a Generative Adversarial Network (GAN) can be utilized as another data augmentation tool. RNN-based GANs suffer from the fact that they cannot effectively model long sequences of data points with irregular temporal relations. To tackle these problems, we introduce TTS-GAN, a transformer-based GAN which can successfully generate realistic synthetic time series data sequences of arbitrary length, similar to the original ones. Both the generator and discriminator networks of the GAN model are built using a pure transformer encoder architecture. We use visualizations to demonstrate the similarity of real and generated time series and a simple classification task that shows how we can use synthetically generated data to augment real data and improve classification accuracy.

---

**Key Idea:**

Transformer GAN generate synthetic time-series data

**The TTS-GAN Architecture** 

![The TTS-GAN Architecture](./images/TTS-GAN.png)

The TTS-GAN model architecture is shown in the upper figure. It contains two main parts, a generator, and a discriminator. Both of them are built based on the transformer encoder architecture. An encoder is a composition of two compound blocks. A multi-head self-attention module constructs the first block and the second block is a feed-forward MLP with GELU activation function. The normalization layer is applied before both of the two blocks and the dropout layer is added after each block. Both blocks employ residual connections. 


**The time series data processing step**

![The time series data processing step](./images/PositionalEncoding.png)

We view a time-series data sequence like an image with a height equal to 1. The number of time-steps is the width of an image, *W*. A time-series sequence can have a single channel or multiple channels, and those can be viewed as the number of channels (RGB) of an image, *C*. So an input sequence can be represented with the matrix of size *(Batch Size, C, 1, W)*. Then we choose a patch size *N* to divide a sequence into *W / N* patches. We then add a soft positional encoding value by the end of each patch, the positional value is learned during model training. Each patch will then have the data shape *(Batch Size, C, 1, (W/N) + 1)* This process is shown in the upper figure.

---

**Repository structures:**

> ./images

Several images of the TTS-GAN project


> ./pre-trained-models

Saved pre-trained GAN model checkpoints


> dataLoader.py

The UniMiB dataset dataLoader used for loading GAN model training/testing data


> LoadRealRunningJumping.py

Load real running and jumping data from UniMiB dataset


> LoadSyntheticRunningJumping.py

Load Synthetic running and jumping data from the pre-trained GAN models


> functions.py

The GAN model training and evaluation functions


> train_GAN.py

The major GAN model training file


> visualizationMetrics.py

The help functions to draw T-SNE and PCA plots


> adamw.py 

The adamw function file


> cfg.py

The parse function used for reading parameters to train_GAN.py file


> JumpingGAN_Train.py

Run this file to start training the Jumping GAN model


> RunningGAN_Train.py

Run this file to start training the Running GAN model


---

**Code Instructions:**


To train the Running data GAN model:
```
python RunningGAN_Train.py
```

To train the Jumping data GAN model:
```
python JumpingGAN_Train.py
```

A simple example of visualizing the similarity between the synthetic running&jumping data and the real running&jumping data:
```
Running&JumpingVisualization.ipynb
```
---
