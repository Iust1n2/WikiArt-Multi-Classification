# WikiArt-Multi-Classification

# Multi-Label Classification with a Custom CNN and ResNet50 in PyTorch

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Custom CNN Model](#custom-cnn-model)
- [ResNet50 Model](#resnet50-model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)

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

- 
