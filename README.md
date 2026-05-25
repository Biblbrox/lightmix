<p align="center">
  <img src="assets/logo.png" alt="LightMix Logo" width="800">
</p>

## Project description and goals
This repository is the playground for ViT acceleration methods for both training and inference. 
We mainly focus on visual models and aim to extend developed techniques for multimodal tasks, such as
lidar/camera fusion architectures. We strive to make ViT models easier to train and run locally on user
devices.

## Training setup
As we use burn framework, setup should be straightforward. Run ```cargo build --release``` for build
and ```cargo run --release``` to run the training process. In order to change network params or dataset
location you should first, create a local config of your experiments (experiments.local.toml)
```bash
cp experiments.toml experiments.local.toml
```
Edit cache dir in the config to your dataset location.


## Speed improvement techniques
We're focusing on optimizing ViT tight-spots such as self-attention, patch embeddings, and 
data hunger.

### Self-attention replacements
Right now, we're testing the following approaches:
- StaticMixer - static permutation matrices applied to token dimension;
- LearnableMixer - learnable (with sinkhorn) permutation matrices applied to token dimension;
- StochasticMixer - replacement of Q and K matrices with their double-stochastic variants;
- StochasticWindowMixer - replacement of Q and K matrices with their double-stochastic variants in addition with window attention.


## Model zoo
For now, we have the following models:
- ViT (conventional implementation);
- EfficientViT;
- Custom FastViT implementation.

Implementations in progress:
- Swin ViT;
- Token-To-Token ViT.


## Benchmarking
In order to run benchmarks, use ```cargo bench```. Also, you should specify run id for your 
benchmark run. For example, for mixing benchmarks:
```bash
LIGHTMIX_RUN_ID=1 cargo bench -p lightmix --bench mixin
```
