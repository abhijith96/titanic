#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:24:44 2018

@author: abraham
"""

import pandas as pd
import numpy as np
import tensorflow as tf  
from sklearn.metrics import explained_variance_score,accuracy_score
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
from tensorflow.python.framework import ops
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,k*mini_batch_size :(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size :(k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def create_placeholders(n_x, n_y):
   
    X = tf.placeholder(tf.float32, [n_x,None])
    Y = tf.placeholder(tf.float32, [n_y,None])
    return X, Y


def initialize_parameters():
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [15,8], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [15,1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [5,15], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [5,1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,5], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer=tf.zeros_initializer())
   
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3" :b3,
                  }
    
    return parameters  


def forward_propagation(X, parameters):
  
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)                                              # A1 = relu(Z1)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                               # Z2 = np.dot(W2, a1) + b2
                                                  # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3 

def compute_cost(Z3, Y):
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0005,
          num_epochs = 10000, minibatch_size = 32, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape
    n_x=n_x-1                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x,n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z2 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z2, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train[1:,:], Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))
        #print(correct_prediction)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#        print ("Train Accuracy:", accuracy.eval({X: X_train[1:,:], Y: Y_train}))
#        print ("Test Accuracy:", accuracy.eval({X: X_test[1:,:], Y: Y_test}))
        
        return parameters   
    X = tf.placeholder(tf.float32, [n_x,None])
def predict(X, parameters):
        n_x=X.shape[0]
        X1=tf.placeholder(tf.float32, [n_x,None])
        Z2=forward_propagation(X1, parameters)
        verdict=tf.sigmoid(Z2)
        with tf.Session() as sess:
            res=sess.run(verdict, feed_dict={X1:X})
        return res
    
    
    
    
df=pd.read_csv('ready.csv').set_index('PassengerId')
df=df.drop(['Unnamed: 0'], axis=1)



X = df[[col for col in df.columns if col != 'Survived']]
X = (X - X.mean()) / (X.max() - X.min())
y=df['Survived']
#dfX_train, dfX_val, dfy_train, dfy_val = train_test_split(X, y, test_size=0.1)  
X_train=X.values[:800,:]
index=np.zeros((800, 1))
for i in range(index.shape[0]):
    index[i][0]=i
X_train=np.append(index, X_train, axis=1)    
X_train=X_train.T
index2=np.zeros((89, 1))
for i in range(0,89):
    index2[i][0]=i+800
X_val=X.values[800:,:]
X_val=np.append(index2, X_val, axis=1)
X_val=X_val.T
y_train=y.values[:800]
y_train=y_train.reshape(1, y_train.shape[0])
y_val=y.values[800:]
y_val=y_val.reshape(1, y_val.shape[0])


print(X_train.shape)
print(y_train.shape)
parameters = model(X_train, y_train, X_val, y_val)
sample_prediction=predict(X_train[1:,:], parameters)
verdict=np.rint(sample_prediction) 
diff=verdict==y_train
correct=np.sum(diff)
train_set_accuracy=correct/y_train.shape[1]
print("Training set accuracy is: "+ str(train_set_accuracy*100))
    
test_prediction=predict(X_val[1:,:], parameters)
test_verdict=np.rint(test_prediction)
test_diff=test_verdict==y_val
test_correct=np.sum(test_diff)
test_set_accuracy=test_correct/y_val.shape[1]
print("Test set accuracy is: "+ str(test_set_accuracy*100))
    


