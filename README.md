# MNIST
neural network demos on the MNIST data set \

#using a simple, non-convolutional neural net on the MNIST data set\
\
"""\
import tensorflow as tf\
#IMPORTING MNIST DATASET (bunch of 28x28 pics o' penciled digitz)\
from tensorflow.examples.tutorials.mnist import input_data\
mnist = input_data.read_data_sets("MNIST_data/", one_hot = 'True')\
"""
\
\
\
runfile('C:/Users/Rob/Documents/SPYDER/mnist_nn.py', wdir='C:/Users/Rob/Documents/SPYDER')\
Extracting MNIST_data/train-images-idx3-ubyte.gz\
Extracting MNIST_data/train-labels-idx1-ubyte.gz\
Extracting MNIST_data/t10k-images-idx3-ubyte.gz\
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz\
Epoch: 1 cost = 0.704\
Epoch: 2 cost = 0.257\
Epoch: 3 cost = 0.195\
Epoch: 4 cost = 0.161\
Epoch: 5 cost = 0.135\
Epoch: 6 cost = 0.117\
Epoch: 7 cost = 0.107\
Epoch: 8 cost = 0.087\
Epoch: 9 cost = 0.080\
Epoch: 10 cost = 0.072\
Current accuracy:  0.9721
