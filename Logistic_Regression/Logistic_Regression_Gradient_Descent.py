# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:43:31 2018

@author: Ashwin Dhakaita
"""

#import the required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#import the required datasets from sklearn's datasets
data = datasets.load_breast_cancer()
X = data['data']
y = data['target']

#transform outputs to an np-array of lists to facilitate vector operations
y = [[i] for i in y ]

#split the dataset into training and test sets
trainX, testX, trainy, testy = train_test_split(X, y)

#convert numpy series objects to numpy arrays
trainX = np.array(trainX)
trainy = np.array(trainy)
testX = np.array(testX)
testy = np.array(testy)

#class logistic regression
class Logistic_Regression:
    
    #constructor function takes epochs(n_runs) and learning rate(alpha) as arguments
    def __init__(self, n_runs, alpha):
        self.n_runs = n_runs
        self.alpha = alpha
    
    #sigmoid function converts output values into probability values
    def sigmoid(self,x):
        x = [i[0] for i in x]
        r = [(1.0)/(1.0 + np.exp(-i)) for i in x]
        r = [ [0] if i<0.5 else [1] for i in r]
        return r
    
    #accepts training data
    def fit(self, trainX, trainy):
        self.trainX = trainX
        self.trainy = trainy
        self.coef = np.zeros((1,trainX.shape[1]))
        self.coef = self.gradient_descent(trainX, trainy, self.n_runs, self.alpha)
        
    #the gradient descent algorithm for optimisation of parameters
    def gradient_descent(self, trainX, trainy, n_runs, alpha):
        beta = self.coef
        n = len(trainy)
        for i in range(self.n_runs):
            x = trainX.dot(beta.transpose())
            fx = self.sigmoid(x)
            loss = fx - trainy
            loss = [i[0] for i in loss]
            grad = alpha*np.dot(trainX.T,loss)*(1/n)
            beta = beta - grad
            
        return beta
    
    #predicts on test data
    def predict(self, testX):
        return self.sigmoid(np.dot(testX, self.coef.T))

#instantiation and prediction on our classifier
reg = Logistic_Regression(300, 0.005)
reg.fit(trainX, trainy)
print(reg.predict(testX))
print(accuracy_score(np.array(reg.predict(testX)),np.array(testy)))

#comparison of our classifier with sklearn's variant
reg = LogisticRegression()
reg.fit(trainX, trainy)
print(reg.predict(testX))
print(accuracy_score(np.array(reg.predict(testX)),np.array(testy)))