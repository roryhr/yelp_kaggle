# Multi-label classification network

This repository contains the code I used for the
[Yelp Kaggle competition](https://www.kaggle.com/c/yelp-restaurant-photo-classification).

My primary motivation is to use the competition as a learning opportunity. I've developed an object-oriented model model which exposes the familiar scikit-learn API. The data munging potion of the code is unit tested -- why not?

Beyond the application of standard development practices to data science, I've implemented the [latest and greatest convolutional network](http://arxiv.org/abs/1512.03385) in Keras.

## Running the network

The competition involves labeling each image from nine possible labels. If we were to one-hot encode this that's `2**9 = 512` classes, which is not far from the 1000 classes in the ImageNet competition. Their deepest network required 11 billion FLOPs and requires many GPUs to train.