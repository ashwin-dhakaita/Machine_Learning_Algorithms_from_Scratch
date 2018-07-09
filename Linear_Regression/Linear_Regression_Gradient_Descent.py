# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:03:32 2018

@author: Ashwin Dhakaita
"""
#import the required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as rmse
import time

#load the boston housing dataset from sklearn datasets
data = datasets.load_boston()
X = data['data']
y = data['target']

#convert np array into an array of lists to facilitate vector operations
y = [[i] for i in y ]

#split the dataset into training and test sets
trainX, testX, trainy, testy = train_test_split(X, y)

#class linear regression
class Linear_Regression:
    
    #constructor function takes epochs(n_runs) and learning rate(alpha) as arguments
    def __init__(self, n_runs, alpha):
        self.n_runs = n_runs
        self.alpha = alpha
    
    #takes inputs on which regressor is to be trained
    def fit(self, trainX, trainy):
        self.trainX = trainX
        self.trainy = trainy
        self.coef = np.zeros((1,trainX.shape[1]))
        self.coef = self.gradient_descent(trainX, trainy, self.n_runs, self.alpha)
    
    #the gradient descent optimisation algorithm
    def gradient_descent(self, trainX, trainy, n_runs, alpha):
        beta = self.coef
        n = len(trainy)
        for i in range(self.n_runs):
            loss = trainX.dot(beta.transpose())
            loss = loss - trainy
            beta = beta - alpha*((np.dot(trainX.T,loss)).T)*(1/n)
            
        return beta
    
    #function which returns predictions based on the regressor fitted on the training data
    def predict(self, testX):
        return np.dot(testX, self.coef.T)

#instantiation and prediction steps of our regressor
reg = Linear_Regression(3000000, 0.00000001)
reg.fit(trainX, trainy)
print(reg.predict(testX))
print(rmse(np.array(reg.predict(testX)),np.array(testy)))