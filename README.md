# Variational Convolutional Autoencoder to extract key features from 3D Heatwave Data

## Overview

This repository contains code for a Variational Convolutional Autoencoder (VAE) designed to reduce the dimensionality of 3D heatwave data. Additionally, it includes the files for the trained model, as well as the code used to process heatwave data and create TensorFlow records (`tf_records`).

Please refer to the paper: ... 

## Variational Convolutional Autoencoder

### Code

The core of this repository is the implementation of a Variational Convolutional Autoencoder for 3D data. The code is organized as follows:

- VAE/utils_3d.py: This file contains the helperfunctions (e.g. plotting).
- VAE/autoencoder_3d_model.py: This file contains the implementation of the Variational Convolutional Autoencoder with a 3D structure.
- VAE/autoencoder_3d_main.py: This file contains the code used to train the model.
  
### Trained Model

The trained model is saved in the following file:

- DATA/MODEL/VAE_L128.h5: This file contains the weights and architecture of the trained Variational Convolutional Autoencoder.

## Heatwave Data Processing

### Code

The repository provides code to convert the heatwave data into TensorFlow records (`tf_records`). The heatwaves are detected using GDBSCAN, please refer to the method section in the paper for the full details.  The code is organized as follows: 

- DATA/HEATWAVES/TF_records.py: This script extracts and processes the heatwave data cubes from the raw KNMI-LENTIS dataset, and generates TensorFlow records for training the Variational Convolutional Autoencoder. Please note that the raw data (KNMI-LENTIS) is needed to obtain the TensorFlow records. 

### Heatwave Data

This repository 

- DATA/HEATWAVES/processed/: This directory contains the csv heatwave data files, as obtained with GDBSCAN, needed to extract the data from the KNMI-LENTIS data. 

