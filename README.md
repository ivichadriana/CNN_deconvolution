Ongoing project comparing neural network inductive biases for image classification under structured and fully shuffled pixel inputs, with matched parameter budgets across architectures.

## Project status

This project currently focuses on comparing:

- Multi-layer perceptrons (MLPs)
- Convolutional neural networks (CNNs)
- Vision Transformer-style models (Transformers)

across image classification tasks using:

- MNIST
- FashionMNIST
- CIFAR10
- PCam support in the utilities pipeline

The current implementation includes:
- model definitions in `src/utils.py`
- Ray Tune-based hyperparameter search in `scripts/tune.py`
- support for intact and fully shuffled image variants
- matched search spaces designed to keep model capacities broadly comparable across architectures

## Main scripts

These scripts are intended to be launched through their corresponding bash or SLURM scripts.

1. `scripts/tune.py`  
   Performs hyperparameter tuning with Ray Tune for MLP, CNN, and Transformer models.

2. `train.py`  
   Trains models using selected hyperparameters.  
   This script may reflect an earlier CNN/MLP workflow and should be checked for consistency with the current Transformer-enabled setup if used.

## Current research goal

The main research goal is to study how architectural inductive bias affects classification performance when model capacity is controlled.

In particular, the project asks whether convolutional structure is beneficial only when spatial structure is present, or whether CNNs can still compete strongly even when spatial relationships are destroyed.

This was extended beyond the original CNN-vs-MLP comparison to also include Transformer-based models.

## Research question

Can CNNs, MLPs, and Transformer models be compared fairly on image classification tasks when their trainable parameter counts are matched as closely as possible?

More specifically:

- Do CNNs outperform MLPs and Transformers on intact images with meaningful spatial structure?
- What happens when the input is fully shuffled and spatial locality is destroyed?
- Do CNNs still retain an advantage due to optimization or representational properties, even when their spatial prior is no longer aligned with the data?
- How do Transformers behave relative to CNNs and MLPs under the same parameter budget constraints?

## Working hypothesis

The working hypothesis is:

- **CNNs** should perform best on intact images because they exploit local spatial structure.
- **MLPs** may become more competitive on fully shuffled inputs because they do not assume locality.
- **Transformers** provide a third comparison point: they process images as patch sequences and may behave differently depending on whether patch structure remains informative after shuffling.

## Fair-comparison design

A major part of this project was designing the comparison to be as fair as possible.

Instead of comparing arbitrary architectures, the search spaces for each model family were constructed so that:

- minimum parameter counts are approximately aligned
- maximum parameter counts are approximately aligned
- model depth is kept broadly comparable across families

### Architectural families used

#### MLP
- 3 hidden layers
- BatchNorm after each hidden layer
- Dropout after each hidden layer
- Final linear classification layer

#### CNN
- 3 convolutional blocks
- BatchNorm after each convolution
- Average pooling after each block
- Dropout after each block
- 1 hidden fully connected layer
- Final classification layer

Two versions are used:
- `SimpleCNN` for 1-channel data
- `SimpleCNN_3CH` for 3-channel data

#### Transformer
- ViT-style patch embedding using a convolutional patch projection
- Learnable class token
- Learnable positional embeddings
- 3 Transformer encoder layers
- LayerNorm before classification head
- Final linear classification head

A single Transformer class is used, with `in_channels` adjusted for 1-channel or 3-channel datasets.

## Why parameter matching matters

A central goal of the project is to compare architectures rather than raw model size.

To make the comparison more meaningful, hyperparameter search spaces were chosen so that each model class can explore configurations within similar trainable-parameter ranges.

This avoids unfair comparisons where one family consistently has far more capacity than another.

## Tuning framework

Hyperparameter tuning is done with **Ray Tune** using:

- random search over architecture-specific parameter spaces
- ASHA scheduling for early stopping
- repeated tuning runs to collect best configurations

### Tuned hyperparameters

Depending on model family, the tuning space includes:

#### MLP
- hidden layer widths
- dropout
- learning rate
- batch size

#### CNN
- convolution channel widths
- hidden fully connected size
- kernel size
- stride
- padding
- dropout
- learning rate
- batch size

#### Transformer
- patch size
- embedding dimension
- number of heads
- Transformer MLP dimension
- dropout
- learning rate
- batch size

For Transformers, additional care was taken to ensure:
- patch sizes remain valid for image dimensions
- embedding dimension is compatible with the number of attention heads

## Data handling

The utilities support both standard and shuffled versions of datasets.

Two shuffling styles exist in the codebase:

- row/column shuffling
- full pixel shuffling

The current tuning workflow uses the full-shuffle loading path:
- `load_training_data_fullshuffle`
- `load_testing_data_fullshuffle`

This is important because the active comparison work moved toward **fully shuffled inputs**, not only row/column permutations.

## Current implementation status

### Implemented and reconstructed
- `src/utils.py`
- `scripts/tune.py`
- MLP, CNN, and Transformer model definitions
- dropout-aligned architectures
- Transformer-compatible tuning setup
- matched search spaces for CNN/MLP/Transformer
- support for shuffled and non-shuffled datasets in the utilities layer

### Explicitly worked on in this phase
- adding a custom Transformer architecture
- matching Transformer search space to CNN/MLP parameter ranges
- updating model constructors to include dropout
- adapting the tuning code to initialize all three architecture families
- debugging cluster launch issues and path issues for SLURM execution

### Not yet fully completed in this phase
- full end-to-end successful Transformer experiment runs
- final result tables and plots
- finalized evaluation writeup across all datasets

## Environment

Conda environment specifications are stored in the `/environment` folder.

Use:

```bash
conda env create -f env_cnn_.yml


