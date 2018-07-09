# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:44:09 2018

@author: Ashwin Dhakaita
"""
#import the requisite modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as accuracy_score

#import the data
data = pd.read_csv('H://winequality-red.csv',sep=';')
X = data.iloc[:,:-1]
y = data['quality']

#split the dataset into training and test sets
trainX, testX, trainy, testy = train_test_split(X,y)

#computation of parameters using noraml equation
beta = (np.linalg.inv((trainX.T).dot(trainX))).dot(trainX.T).dot(trainy)
print(beta)

#rendering predictions
pred = (np.dot(testX,beta))
print(accuracy_score(testy,pd.Series(pred)))

#testing our regressor with the sklearn's variant
reg = LinearRegression()
reg.fit(trainX,trainy)
pred = reg.predict(testX)
print(reg.coef_)
print(accuracy_score(pred,testy))