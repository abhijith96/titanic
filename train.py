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



def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):  
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)
    
    
#df=pd.read_csv('ready.csv').set_index('PassengerId')
#df=df.drop(['Unnamed: 0'], axis=1)
#X = df[[col for col in df.columns if col != 'Survived']]
#y=df['Survived']
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=23)  
#feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]  
#
#classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols,  
#                                      hidden_units=[10],
#                                      optimizer=tf.train.ProximalAdagradOptimizer(
#                                      learning_rate=0.004,
#                                      l1_regularization_strength=0.0001))
#
#evaluations = []  
#STEPS = 50
#for i in range(50):  
#    classifier.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
#    evaluations.append(classifier.evaluate(input_fn=wx_input_fn(X_val,
#                                                               y_val,
#                                                               num_epochs=1,
#                                                               shuffle=False)))
#    
#    
#    
#    
#plt.rcParams['figure.figsize'] = [14, 10]
#
#loss_values = [ev['loss'] for ev in evaluations]  
#training_steps = [ev['global_step'] for ev in evaluations]
#
#plt.scatter(x=training_steps, y=loss_values)  
#plt.xlabel('Training steps (Epochs = steps / 2)')  
#plt.ylabel('Loss (SSE)')  
#plt.show()      

#
#df2=pd.read_csv('ready_test.csv').set_index('PassengerId')
##df2=df2.drop(['Unnamed: 0'], axis=1)
#X_test = df2[[col for col in df2.columns if col != 'Survived']]
#y_test=df2['Survived']
##
#
pred = classifier.predict(input_fn=wx_input_fn(X_test,  
                                              num_epochs=1,
                                              shuffle=False))

print(list(pred)[0])
pred2 = classifier.predict(input_fn=wx_input_fn(X_train,  
                                              num_epochs=1,
                                              shuffle=False))

test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=X_test.values,
      y=y_test.values,
      num_epochs=1,
      shuffle=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


print("The training set accuracy: %.2f" % accuracy_score(  
                                            y_train, pred3))  

#print("The Explained Variance: %.2f" % explained_variance_score(  
#                                            y_test, predictions))  
#print("The test set accuracy: %.2f" % accuracy_score(  
#                                            y_test, predictions))  
#
#    
