
# coding: utf-8

# # Using slim to Build Model Architectures
# 
# This notebook gives a simple example of using Tensorflow's [slim library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) to construct a "deeper" learning architecture for the MNIST dataset. We'll see how easy it is to construct architectures, and how to output simple summaries from Tensorboard.
# 
# As a simple example, we'll use the architecture presented in Michael Nielsen's online [textbook](http://neuralnetworksanddeeplearning.com/chap6.html). This knowledge will later become useful as we explore more complicated architectures like [Inception](https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py).

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data


# In[2]:


# Read in MNIST dataset, compute mean / standard deviation of the training images
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)

MEAN = np.mean(mnist.train.images)
STD = np.std(mnist.train.images)


# In[3]:


# Convenience method for reshaping images. The included MNIST dataset stores images
# as Nx784 row vectors. This method reshapes the inputs into Nx28x28x1 images that are
# better suited for convolution operations and rescales the inputs so they have a
# mean of 0 and unit variance.
def resize_images(images):
    reshaped = (images - MEAN)/STD
    reshaped = np.reshape(reshaped, [-1, 28, 28, 1])
    
    assert(reshaped.shape[1] == 28)
    assert(reshaped.shape[2] == 28)
    assert(reshaped.shape[3] == 1)
    
    return reshaped


# ## NielsenNet
# 
# The neural net architecture presented by Michael Nielsen in chapter 6 of his textbook achieves an accuracy in excess of 99%. I've dubbed this architecture _NielsenNet_. It consists of two convolution layers, followed by two fully connected neural network layers, followed by an output layer. Dropout is used to after each fully-connected layer to control overfitting.
# 
# When building this model using slim, we'll use the built-in `conv2d` and `max_pool` functions to build the convlution layers, changing 28x28 input images into 5x5x40 outputs for the fully connected layer. We'll do this with a combination of different padding modes (`SAME` vs `VALID`) and max-pooling.
# 
# Finally we'll build a succession of fully-connected layers using the `fully_connected` convenience method. Dropout can be implemented using slim's `dropout` method, gated by the `is_training` tensor.
# 
# Building the network by successively mutating the `net` variable is pretty common, and something we'll see later on in the Inception architecture ([peek here](https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py#L107)).

# In[4]:


def nielsen_net(inputs, is_training, scope='NielsenNet'):
    with tf.variable_scope(scope, 'NielsenNet'):
        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 5*5*40])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        # Output Layer: 1000x1 => 10x1
        net = slim.fully_connected(net, 10, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net


# In[5]:


sess = tf.InteractiveSession()

# Create the placeholder tensors for the input images (x), the training labels (y_actual)
# and whether or not dropout is active (is_training)
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')
is_training = tf.placeholder(tf.bool, name='IsTraining')

# Pass the inputs into nielsen_net, outputting the logits
logits = nielsen_net(x, is_training, scope='NielsenNetTrain')


# In[6]:


# Use the logits to create four additional operations:
#
# 1: The cross entropy of the predictions vs. the actual labels
# 2: The number of correct predictions
# 3: The accuracy given the number of correct predictions
# 4: The update step, using the MomentumOptimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=logits))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)


# In[7]:


# To monitor our progress using tensorboard, create two summary operations
# to track the loss and the accuracy
loss_summary = tf.summary.scalar('loss', cross_entropy)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('/tmp/nielsen-net', sess.graph)


# In[8]:


eval_data = {
    x: resize_images(mnist.validation.images),
    y_actual: mnist.validation.labels,
    is_training: False
}

for i in range(2000):
    images, labels = mnist.train.next_batch(100)
    summary, _ = sess.run([loss_summary, train_step], feed_dict={x: resize_images(images), y_actual: labels, is_training: True})
    train_writer.add_summary(summary, i)
    
    if i % 200 == 0:
        summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval_data)
        train_writer.add_summary(summary, i)
        print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))


# In[9]:


# test_data = {
#     x: resize_images(mnist.test.images),
#     y_actual: mnist.test.labels,
#     is_training: False
# }

# Due to memory limitation, and on the basis of good running above,
# sample number small than that of eval_data (that is 5500) will be ok.
# While, the number size of test_data is 10000, so I choose 5000, 
# and do a 5-loop repetition.
for i in range(5):
    images, labels = mnist.test.next_batch(5000)
    summary, acc = sess.run([accuracy_summary, accuracy], feed_dict={x: resize_images(images), y_actual: labels, is_training: False})
    print("Step: %5d, Test Accuracy = %5.2f%%" % (i, 100 * acc))