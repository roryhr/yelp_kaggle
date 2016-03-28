# Multi-label classification network

This repository contains the code I used for the
[Yelp Kaggle competition](https://www.kaggle.com/c/yelp-restaurant-photo-classification).

My primary motivation is to use the competition as a learning opportunity. I've developed an object-oriented model model which exposes the familiar scikit-learn API. The data munging potion of the code is unit tested -- why not?

Beyond the application of standard development practices to data science, I've implemented the [latest and greatest convolutional network](http://arxiv.org/abs/1512.03385) in Keras.

## Running the network

The competition involves labeling each image from nine possible labels. If we were to one-hot encode you'd have `2**9 = 512` classes, not far from the 1000 classes in the ImageNet competition. Their deepest network required 11 billion FLOPs and a lot of time and resources to train. Consider yourself warned. 


# What's the model look like? 

Glad you asked! Here's the *34-layer* residual model. 

```
>>> model = KerasGraphModel()
>>> model.build_residual_network()
>>> model.graph.summary()
--------------------------------------------------------------------------------
Layer (name)                  Output Shape                  Param #
--------------------------------------------------------------------------------
Layer (input)                 (None, 3, 224, 224)           0
Convolution2D (conv1_1)       (None, 64, 112, 112)          9472
BatchNormalization (bn1_1)    (None, 64, 112, 112)          224
Activation (relu1_1)          (None, 64, 112, 112)          0
MaxPooling2D (pool1)          (None, 64, 56, 56)            0
Convolution2D (conv2_1)       (None, 64, 56, 56)            36928
BatchNormalization (bn2_1)    (None, 64, 56, 56)            112
Activation (relu2_1)          (None, 64, 56, 56)            0
Convolution2D (conv2_2)       (None, 64, 56, 56)            36928
BatchNormalization (bn2_2)    (None, 64, 56, 56)            112
Activation (relu2_2)          (None, 64, 56, 56)            0
Convolution2D (conv2_3)       (None, 64, 56, 56)            36928
BatchNormalization (bn2_3)    (None, 64, 56, 56)            112
Activation (relu2_3)          (None, 64, 56, 56)            0
Convolution2D (conv2_4)       (None, 64, 56, 56)            36928
BatchNormalization (bn2_4)    (None, 64, 56, 56)            112
Activation (relu2_4)          (None, 64, 56, 56)            0
Convolution2D (conv2_5)       (None, 64, 56, 56)            36928
BatchNormalization (bn2_5)    (None, 64, 56, 56)            112
Activation (relu2_5)          (None, 64, 56, 56)            0
Convolution2D (conv2_6)       (None, 64, 56, 56)            36928
BatchNormalization (bn2_6)    (None, 64, 56, 56)            112
Activation (relu2_6)          (None, 64, 56, 56)            0
Convolution2D (conv3_1)       (None, 128, 28, 28)           73856
BatchNormalization (bn3_1)    (None, 128, 28, 28)           56
Activation (relu3_1)          (None, 128, 28, 28)           0
Convolution2D (conv3_2)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_2)    (None, 128, 28, 28)           56
Convolution2D (short3_1)      (None, 32, 56, 56)            2080
BatchNormalization (short_bn3_(None, 32, 56, 56)            112
Reshape (short_reshape3_1)    (None, 128, 28, 28)           0
Activation (relu3_2)          (None, 128, 28, 28)           0
Convolution2D (conv3_3)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_3)    (None, 128, 28, 28)           56
Activation (relu3_3)          (None, 128, 28, 28)           0
Convolution2D (conv3_4)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_4)    (None, 128, 28, 28)           56
Activation (relu3_4)          (None, 128, 28, 28)           0
Convolution2D (conv3_5)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_5)    (None, 128, 28, 28)           56
Activation (relu3_5)          (None, 128, 28, 28)           0
Convolution2D (conv3_6)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_6)    (None, 128, 28, 28)           56
Activation (relu3_6)          (None, 128, 28, 28)           0
Convolution2D (conv3_7)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_7)    (None, 128, 28, 28)           56
Activation (relu3_7)          (None, 128, 28, 28)           0
Convolution2D (conv3_8)       (None, 128, 28, 28)           147584
BatchNormalization (bn3_8)    (None, 128, 28, 28)           56
Activation (relu3_8)          (None, 128, 28, 28)           0
Convolution2D (conv4_1)       (None, 256, 14, 14)           295168
BatchNormalization (bn4_1)    (None, 256, 14, 14)           28
Activation (relu4_1)          (None, 256, 14, 14)           0
Convolution2D (conv4_2)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_2)    (None, 256, 14, 14)           28
Convolution2D (short4_1)      (None, 64, 28, 28)            8256
BatchNormalization (short_bn4_(None, 64, 28, 28)            56
Reshape (short_reshape4_1)    (None, 256, 14, 14)           0
Activation (relu4_2)          (None, 256, 14, 14)           0
Convolution2D (conv4_3)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_3)    (None, 256, 14, 14)           28
Activation (relu4_3)          (None, 256, 14, 14)           0
Convolution2D (conv4_4)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_4)    (None, 256, 14, 14)           28
Activation (relu4_4)          (None, 256, 14, 14)           0
Convolution2D (conv4_5)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_5)    (None, 256, 14, 14)           28
Activation (relu4_5)          (None, 256, 14, 14)           0
Convolution2D (conv4_6)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_6)    (None, 256, 14, 14)           28
Activation (relu4_6)          (None, 256, 14, 14)           0
Convolution2D (conv4_7)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_7)    (None, 256, 14, 14)           28
Activation (relu4_7)          (None, 256, 14, 14)           0
Convolution2D (conv4_8)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_8)    (None, 256, 14, 14)           28
Activation (relu4_8)          (None, 256, 14, 14)           0
Convolution2D (conv4_9)       (None, 256, 14, 14)           590080
BatchNormalization (bn4_9)    (None, 256, 14, 14)           28
Activation (relu4_9)          (None, 256, 14, 14)           0
Convolution2D (conv4_10)      (None, 256, 14, 14)           590080
BatchNormalization (bn4_10)   (None, 256, 14, 14)           28
Activation (relu4_10)         (None, 256, 14, 14)           0
Convolution2D (conv4_11)      (None, 256, 14, 14)           590080
BatchNormalization (bn4_11)   (None, 256, 14, 14)           28
Activation (relu4_11)         (None, 256, 14, 14)           0
Convolution2D (conv4_12)      (None, 256, 14, 14)           590080
BatchNormalization (bn4_12)   (None, 256, 14, 14)           28
Activation (relu4_12)         (None, 256, 14, 14)           0
Convolution2D (conv5_1)       (None, 512, 7, 7)             1180160
BatchNormalization (bn5_1)    (None, 512, 7, 7)             14
Activation (relu5_1)          (None, 512, 7, 7)             0
Convolution2D (conv5_2)       (None, 512, 7, 7)             2359808
BatchNormalization (bn5_2)    (None, 512, 7, 7)             14
Convolution2D (short5_1)      (None, 128, 14, 14)           32896
BatchNormalization (short_bn5_(None, 128, 14, 14)           28
Reshape (short_reshape5_1)    (None, 512, 7, 7)             0
Activation (relu5_2)          (None, 512, 7, 7)             0
Convolution2D (conv5_3)       (None, 512, 7, 7)             2359808
BatchNormalization (bn5_3)    (None, 512, 7, 7)             14
Activation (relu5_3)          (None, 512, 7, 7)             0
Convolution2D (conv5_4)       (None, 512, 7, 7)             2359808
BatchNormalization (bn5_4)    (None, 512, 7, 7)             14
Activation (relu5_4)          (None, 512, 7, 7)             0
Convolution2D (conv5_5)       (None, 512, 7, 7)             2359808
BatchNormalization (bn5_5)    (None, 512, 7, 7)             14
Activation (relu5_5)          (None, 512, 7, 7)             0
Convolution2D (conv5_6)       (None, 512, 7, 7)             2359808
BatchNormalization (bn5_6)    (None, 512, 7, 7)             14
Activation (relu5_6)          (None, 512, 7, 7)             0
AveragePooling2D (pool2)      (None, 512, 1, 1)             0
Flatten (flatten)             (None, 512)                   0
Dense (dense)                 (None, 9)                     4617
Dense (output)                (None, 9)                     4617
--------------------------------------------------------------------------------
Total params: 21153041
--------------------------------------------------------------------------------

```