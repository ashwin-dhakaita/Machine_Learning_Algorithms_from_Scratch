# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 12:19:37 2018

@author: Ashwin Dhakaita
"""
#import the required modules
import numpy as np

#create datasets
X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y = np.array([[0], [1], [1], [0]])

#nonlin function computes both sigmoid and derivative of sigmoid activation function
def nonlin(x, deriv=False):
    if deriv==True:
        return x*(1-x)
    return (1.0)/(1.0 + np.exp(-x))

#initialize the synapses with random weights
w1 = 2*np.random.random(size=(3,7)) - 1
w2 = 2*np.random.random(size=(7,1)) - 1

#assign iteration_rounds (epochs) and learning rate of the neural network
epochs = 60000
lr = 1

#neural network being trained by gradient descent
for i in range(epochs):
    
    #feedforwarding the neural network
    l0 = X
    l1 = np.dot(l0,w1)
    l2 = np.dot(l1,w2)
    
    #computing the activations
    z1 = nonlin(l1)
    z2 = nonlin(l2)
    
    #backpropagation to fit the required weights
    error_output = y - z2
    d_output = error_output * nonlin(z2, deriv=True)
    error_hidden = d_output.dot(w2.T)
    d_hidden = error_hidden * nonlin(z1, deriv=True)
    
    #gradient descent update rules
    w1 += (X.T).dot(d_hidden)*lr
    w2 += (z1.T).dot(d_output)*lr

#rendering the final output
print(np.round(z2))