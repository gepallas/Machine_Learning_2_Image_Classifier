# Image Classifier trained to recognize species of flowers
This project has been completed as part of the [Udacity's Machine Learning Nanodegree](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229) requirements.

## Overview
In this project an image classifier is trained using deep learning to recognize different species of flowers.

The projet is split in two parts:
- Development of the ***Image Classifier*** to recognize flower species: *Image Classifier Project.ipynb* file
- ***Command Line Application*** development that others can use: *train.py* and *predict.py* files

A ***pre-trained network*** from torchvision.models has been used to get the image features (*model transfer*). 
Specifically, the steps that have been taken to use the pre-trained network are:
- Load a pre-trained network
- Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
- Train the classifier layers using backpropagation using the pre-trained network to get the features
- Track the loss and accuracy on the validation set to determine the best hyperparameters

The project steps can be seen in detail in the jupyter notebook file:
- Data load and pre-processing
- Building and training the classifier
- Testing the classifier
- Save/Load a checkpoint for the trained classifier
- Inference for classification (image pre-processing, class prediction)
- Sanity test (show image and plot the probabilities for the top 5 classes)

## Requirements
- Python
- Libaries: Torch, Pandas, NumPy, MatPlotlib, Seaborn, PIL, json
- Jupyter Notebook

## Results
The model achieves about 75% prediction accuracy on the testset.
However, since the trained model checkpoint is saved, it can be loaded and further trained if one wants to achieve higher accuracy.

**Image prediction and plotting of the probabilites for the top 5 flower classes:**

<p align="center">
  <img src= "https://github.com/gepallas/Machine_Learning_2_Image_Classifier/blob/master/rose_image_prediction.png?raw=true" />
</p>
