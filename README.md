# Wafer Anomaly Detection using Autoencoders
An implementation of feedforward and LSTM-based autoencoders in Python for anomaly detection in semiconductor manufacturing timeseries data.

## Overview
This project uses deep learning autoencoders to learn the patterns of normal wafer from timeseries sensor data. The model is trained only on normal data to reconstruct it with low error. When an anomalous sample is introduced, the model produces a high reconstruction error, allowing for effective anomaly detection by setting an error threshold.

## Dataset
The Wafer dataset consists of 152-length recordings. It includes 1000 training samples and 6164 test samples.

Source: You can find the dataset and more information at [UCR Time Series Classification Archive](https://www.timeseriesclassification.com/dataset.php).

## Models Implemented
This project provides two autoencoder implementations:

- Dense Autoencoder: A standard feedforward neural network.

- LSTM Autoencoder: A recurrent neural network ideal for sequential data.
