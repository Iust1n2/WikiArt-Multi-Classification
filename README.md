# WikiArt-Multi-Classification

## Multi-Label Classification with a Custom CNN and ResNet50 in PyTorch

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Custom CNN Model](#custom-cnn-model)
- [ResNet50 Model](#resnet50-model)
- [Training](#training)
- [Evaluation](#evaluation)

## Introduction

Multi-Class Classification of paintings curated from WikiArt using a Custom Convolutional Network and a ResNet50 pre-trained model in PyTorch. 

The objective is to classify images based on 3 labels: artist, style and genre and by doing so we need to create in our Neural Networks 3 fully connected layers that perform this task.

## Requirements

Ensure the following dependencies are installed:


| Dependency      | Version |
|-----------------|---------|
| Python          | 3.9.18     |
| PyTorch         | 2.1.1+cu121   |
| Torchvision     | 0.16.1+cu121    |
| Datasets (HuggingFace) |  2.12.0   |

## Dataset 

The dataset consists of  81,444 images of paintings along with class labels for each image: 
- artist : 129 artist classes, including a "Unknown Artist" class
- genre : 11 genre classes, including a "Unknown Genre" class
- style : 27 style classes

Link to dataset: https://huggingface.co/datasets/huggan/wikiart

## Custom CNN Model

WikiArtModel is a custom Convolutional Neural Network suited for multi-label classification.

1. **Shared Convolutional Layers**: The model starts with three convolutional layers (conv1, conv2, conv3) with `batch normalization` (bn1, bn2, bn3) and `ReLU activation functions`. These layers are responsible for extracting features from the input images.

2. **Max-Pooling**: After each convolutional layer, max-pooling is applied to reduce the spatial dimensions of the feature maps.

3. **Fully Connected Layers**: Following the convolutional layers, there are three separate branches for artist classification, genre classification, and style classification.

  * `Artist Classification Branch`: This branch includes a `fully connected layer` (fc_artist1), `batch normalization` (bn_artist), `dropout` (dropout_artist), and another `fully connected layer` (fc_artist2) to predict the artist label.

  * `Genre Classification Branch`: Similar to the artist branch, this branch has a `fully connected layer` (fc_genre1), `batch normalization` (bn_genre), `dropout` (dropout_genre), and a `fully connected layer` (fc_genre2) to predict the genre label.

  * `Style Classification Branch`: This branch follows the same structure as the artist and genre branches, with a `fully connected layer` (fc_style1), `batch normalization` (bn_style), `dropout` (dropout_style), and a fully connected layer (fc_style2) to predict the style label.

4. **Forward Pass**: In the forward method, the input image is passed through the shared convolutional layers, followed by each classification branch separately. ReLU activations, batch normalization, and dropout are applied as specified in each branch.

### Results

The WikiArtModel custom CNN struggled with the big number of parameters: `308,743,335` and a Estimated Total Size: `1325.52 MB` and at around Epoch 40, in the forth training loop, using a lr of 0.001 and with Dropout and Batch Norm it showed signs of overfitting so the process was not continued.


## ResNet50 Model

ResNet50 is a deep Residual Neural Network that comes with pre-trained weights from ImageNet1000. To make it work with our task we need to modify it's internal architecture to make classification based on our labels and their coresponding classes.  It consists of 50 layers and is part of the ResNet (Residual Network) family of models.

ResNet50, the model that was used for this task uses a Deeper Bottleneck Architecture : `"ResNet50: the 2-layer skip block in the 34-layer Net is replaced with a 3-layer bottleneck block, resulting in a 50-layer ResNet."`, as the authors Kaiming He et. al. describe in their paper from 2017 - *Deep Residual Learning for Image Classification*. 

The `3-layer bottleneck block` is different from a simple residual block because in addition to it, the bottleneck block uses:

*  **Input Transformation**: A `1x1 convolutional layer`, often reffered to as the `bottleneck layer`. The purpose of this layer is to reduce the dimensionality of the feature maps, which helps reduce computational complexity
*  **Intermediate Transformation**: A `3x3 convolutional layer`, in which the feature maps resulting from the Input Transformation are passed through. This layer is responsible for capturing complex spatial patterns in the data, because it uses a larger number of filters than the previous convolution.
* **Output Transformation**: The output of the 3x3 convolution is then passed through another `1x1 convolutional layer` with a reduced number of filters. This final 1x1 convolution helps restore the dimensionality of the feature maps to match the input dimensions before the addition operation.

* **Skip (Shorctcut) Connection**: In parallel with the above transformations, the input to the block (identity mapping) is passed directly through a `1x1 convolutional layer`. The shortcut connection ensures that the gradient can flow freely through the block, making backpropagation smoother in these large (deep) networks, facilitating training.

### Results

Transfer Learning with ResNet50 in our case proves to be more efficient in dealing with this rather complex classification, due to the fact that the network is attempting to learn representations of almost 200 classes. ResNet is great for these kind of tasks because it was pre-trained on ImageNet 1000, a large benchmark dataset comprised of 1 million images of 1000 classes.

In the Inference part of the notebook, our ResNet model managed to predict correctly the first two labels of a Monet with a `70% accuracy`, the first corresponding to the artist, which in our case is `Claude Monet` and the second label, genre as `Impressionism`.

## Training 

There were notable differences in each of the model's training process, namely WikiArt incrementally improved the loss and accuracy over 40 epochs, which based on the size and complexity of our data took 18 hours to train, only to achieve a `49.6% accuracy` and a loss of `1.6711`.

ResNet50, however, after 46 epochs, which is almost 19 hours, the accuracy and loss improved significantly: `72% accuracy` and `0.8126` loss.

## Evaluation

Same as in Training, each model performed differently, WikiArt reached a final validation accuracy of `52%` and `1.57` loss.

ResNet reached a final validation accuracy of `69.95%` and a loss of `0.9311`.
