# Introduction #
MNIST ("Modified National Institute of Standards and Technology") is the de facto “Hello World” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

# Problem Statement #
In this, we aim to correctly identify digits from a dataset of tens of thousands of handwritten images.

## Hardware ##
1. 940MX GPU with CUDA 10.1
2. 8GB RAM at 2133 Hz.
3. i5 @2.4 Ghz with 4 cores.

## Process ##
The process is divided into 3 parts.
1. Add __Gaussian noise__ to dataset.
2. Create __Autoencoder__ to denoise the image.
3. Create a __Dense layer model__ to identify Digits.


### Why Gaussian Noise ###
- It helps to prevent GAN attacks.
- It can be used for dimensionality reduction.


1. ## Adding Gaussian Noise ##
    1. [How to add gaussian noise](https://blog.keras.io/building-autoencoders-in-keras.html)
    2. Here is the link to my the code. [Link](https://github.com/Ashish-Surve/Machine-Learning/blob/master/MNIST_AutoEncoder/MNIST_CNN.py)

2. ## Create Autoencoder ##
    1. __Encoder__ - Input image => CONV => RELU => BN => CONV => RELU => BN => Flatten => Dense
    2. __Decoder__ - Output Encoder image => CONV_TRANSPOSE => RELU => BN =>  CONV_TRANSPOSE => RELU => BN => CONV_TRANSPOSE => Sigmoid
    3. __Autoencoder__ - Encoder + Decoder

3. ## Dense Layers model for prediction ##
    1. Dense => Dropout => Dense => Dropout => Dense => Dropout => Dense => Dense
    2. Output one hot encoded numbers between 0-9
    
Property                |Value 
------------------------|---------
Epochs(Autoencoder)     |100      
Epochs(Dense)           |200      
Optimizer(Autoencoder)     |Adam   
Optimizer(Dense)           |Adam    

    
Property                |Training | Testing
------------------------|---------|---------
No. of Images           |60,000   | 10,000
Time (Autoencoder)      |2 hrs    | 23 mins 
Time (Digit Prediction) |20 mins  | 4 mins


## Loss vs Epoch for Autoencoder
![Chart1](https://raw.githubusercontent.com/Ashish-Surve/Machine-Learning/master/MNIST_AutoEncoder/plot.png)

## Loss vs Epoch for Predictor
![Chart2](https://raw.githubusercontent.com/Ashish-Surve/Machine-Learning/master/MNIST_AutoEncoder/Dense_plot.png)






    
