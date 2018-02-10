# MNIST
neural network demos on the MNIST data set \

#using tensorflow and a 3 layer NN (non-convolutional)\
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
Test accuracy:  0.9721\
\
\
#using a CNN with Keras\
\
x_train shape: (60000, 28, 28, 1)\
60000 train samples\
10000 test samples\
Train on 60000 samples, validate on 10000 samples\
Epoch 1/10\
60000/60000 [==============================] - 106s 2ms/step - loss: 0.1518 - acc: 0.9533 - val_loss: 0.0429 - val_acc: 0.9861\
Epoch 2/10\
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0407 - acc: 0.9877 - val_loss: 0.0355 - val_acc: 0.9889\
Epoch 3/10\
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0263 - acc: 0.9917 - val_loss: 0.0297 - val_acc: 0.9901\
Epoch 4/10\
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0207 - acc: 0.9936 - val_loss: 0.0262 - val_acc: 0.9922\
Epoch 5/10\
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0157 - acc: 0.9951 - val_loss: 0.0256 - val_acc: 0.9922\
Epoch 6/10\
60000/60000 [==============================] - 105s 2ms/step - loss: 0.0107 - acc: 0.9966 - val_loss: 0.0375 - val_acc: 0.9882\
Epoch 7/10\
60000/60000 [==============================] - 109s 2ms/step - loss: 0.0113 - acc: 0.9965 - val_loss: 0.0354 - val_acc: 0.9893\
Epoch 8/10\
60000/60000 [==============================] - 105s 2ms/step - loss: 0.0088 - acc: 0.9970 - val_loss: 0.0304 - val_acc: 0.9909\
Epoch 9/10\
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0089 - acc: 0.9970 - val_loss: 0.0283 - val_acc: 0.9919\
Epoch 10/10\
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0075 - acc: 0.9975 - val_loss: 0.0400 - val_acc: 0.9913\
Test loss: 0.03996784271821889\
Test accuracy: 0.9913\
\
\
#CNN with tensorflow not operational :(
