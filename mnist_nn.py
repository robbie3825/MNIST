# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:51:14 2018
@author: Rob

"""
#using a simple, non-convolutional neural net on the MNIST data set

import tensorflow as tf

#IMPORTING MNIST DATASET (bunch of 28x28 pics o' penciled digitz)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = 'True')


#Python optimization parameters
learning_rate = 0.5
epochs = 10
batch_size = 100

#declare the training-data placeholders
#input x - for 28 x 28 pixels (size is 784)
x = tf.placeholder(tf.float32, [None, 784])
#output, 10 digits
y = tf.placeholder(tf.float32, [None,10])

#number of layers is 3, so we need 3-1 weight/bias tensors
#these weights connect the input to the hidden
W1 = tf.Variable(tf.random_normal([784,300], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([300]), name = 'b1')

#these connect the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300,10], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([10]), name = 'b2')

#calculate the inpput of the hidden layer
hidden_out = tf.add(tf.matmul(x,W1), b1)
hidden_out = tf.nn.relu(hidden_out)

#calculate the output of the hidden layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))

#clip y_ in (0,1)
y_clipped = tf.clip_by_value(y_,1e-10,0.9999999)
#funny business (cost function actually)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis = 1))

#add an optimiser to do the gradient descent
optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)


#big boy 'initialization operator'
init_op = tf.global_variables_initializer()

#define an 'accuarcy assessment operation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print('accuracy achieved: ' + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
    