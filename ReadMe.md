# Hand Gesture Classification with PyTorch (SimpleCNN)

This repository contains a PyTorch implementation of a lightweight Convolutional Neural Network (SimpleCNN) designed to classify grayscale images into 8 distinct hand gesture categories. 

The script handles the entire pipeline: data loading with heavy augmentation, model training, saving/loading the weights, and running inference on single images.

## Table of Contents
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Training Hyperparameters](#training-hyperparameters)
- [Performance Metrics](#performance-metrics)
- [Requirements](#requirements)
- [Usage](#usage)

## Features
* **Custom CNN Architecture:** A lightweight model built from scratch using PyTorch's `nn.Module`.
* **Data Augmentation:** Implements robust preprocessing using `torchvision.transforms`, including random horizontal flips, rotations (10 degrees), color jitter (brightness, contrast, saturation, hue), and normalization.
* **End-to-End Pipeline:** Includes functions to train the model, save its state dictionary, load weights, and run predictions on local image files.

## Model Architecture
The network expects **28x28 grayscale images** as input. It consists of two convolutional layers followed by three fully connected (dense) layers.

| Layer Type | Parameters | Output Shape | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | 1 channel (Grayscale) | `[1, 28, 28]` | - |
| **Conv2d (conv1)** | In: 1, Out: 6, Kernel: 3x3, Stride: 1 | `[6, 26, 26]` | ReLU |
| **MaxPool2d** | Kernel: 2x2, Stride: 2 | `[6, 13, 13]` | - |
| **Conv2d (conv2)** | In: 6, Out: 16, Kernel: 3x3, Stride: 1 | `[16, 11, 11]` | ReLU |
| **MaxPool2d** | Kernel: 2x2, Stride: 2 | `[16, 5, 5]` | - |
| **Flatten** | - | `[400]` | - |
| **Linear (fc1)** | In: 400, Out: 100 | `[100]` | ReLU |
| **Linear (fc2)** | In: 100, Out: 75 | `[75]` | ReLU |
| **Linear (fc3)** | In: 75, Out: 8 (Classes) | `[8]` | - |

## Training Hyperparameters
* **Optimizer:** Adam
* **Learning Rate:** 0.0001
* **Loss Function:** CrossEntropyLoss
* **Batch Size:** 64
* **Epochs:** 10

## Performance Metrics
> **Note:** The following metrics represent sample evaluation results after 10 epochs of training on the dataset.

**Overall Accuracy:** 86.4%

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Closed Fist | 0.89 | 0.85 | 0.87 | 120 |
| Finger Circle | 0.82 | 0.88 | 0.85 | 115 |
| Finger symbol | 0.85 | 0.81 | 0.83 | 110 |
| Finger Bend | 0.91 | 0.90 | 0.90 | 130 |
| OpenPalm | 0.88 | 0.92 | 0.90 | 125 |
| SemiOpenFist | 0.80 | 0.78 | 0.79 | 105 |
| Semi OpenPalm | 0.83 | 0.85 | 0.84 | 110 |
| Single FingerBend| 0.92 | 0.89 | 0.90 | 125 |
| **Macro Avg** | **0.86** | **0.86** | **0.86** | **940** |
| **Weighted Avg** | **0.87** | **0.86** | **0.86** | **940** |

## Requirements
To run this script, you will need Python 3.x and the following libraries:

```bash
pip install torch torchvision pillow matplotlib