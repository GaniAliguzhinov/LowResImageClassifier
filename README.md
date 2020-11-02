# LowResImageClassifier
Image classifier for CIFAR-10 and MNIST type of data

Results with 28-8-4 (varying FC layer size, no dropout):

    * 224: (0.0088, 0.9977), (0.0269, 0.9953)

    * 256: (?, ?), (0.0261, 0.9957)

 Optimal filter sizes: 5 for convolution 32, 5 for convolution 64, 3 for conv128.
 Optimal Number of convolutions: 2 for convolution 32, 2 for convolution 64, 1 for conv128.
    
# References



https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook

https://www.tensorflow.org/tutorials/quickstart/beginner

https://www.kaggle.com/poonaml/deep-neural-network-keras-way

https://nickcdryan.com/2017/06/13/dropconnect-implementation-in-python-and-tensorflow/

http://proceedings.mlr.press/v28/wan13.pdf

https://www.tensorflow.org/tutorials/keras/keras_tuner

https://www.kaggle.com/yadavsarthak/residual-networks-and-mnist
