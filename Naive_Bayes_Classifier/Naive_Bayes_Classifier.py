# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:05:56 2018

@author: Ashwin Dhakaita
"""

#import the required modules
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#import the iris dataset from sklearn's datasets
data = datasets.load_iris()
X = data['data']
y = data['target']

#split the dataset into training and test sets
trainX, testX, trainy, testy = train_test_split(X,y)

#convert numpy series object into numpy arrays
trainX = np.array(trainX)
trainy = np.array(trainy)
testX = np.array(testX)
testy = np.array(testy)

#class Naive_Bayes
class Naive_Bayes:
    def fit(self, X, y):
        
        self.X = X
        self.y = y
    
    #function to compute likelihood    
    def likelihood(self,dat, t):
        s = 1
        
        for i in range(len(dat[0])):
            s = s*(1/(2*(np.pi)*np.var(dat[:,i]))**0.5)*(np.exp(-((t[i]-np.mean(dat[:,i]))**2)/(2*np.var(dat[:,i]))))
            
        return s
    
    #function to train the naive bayes algorithm on the given data    
    def train(self,X,y,t):
        l = list(np.unique(y))
        count = np.zeros((1,len(l)))
        dat = []
        
        for i in range(len(l)):
            dat.append([])
            
        for i in range(len(y)):
            count[0,l.index(y[i])] += 1
            dat[l.index(y[i])].append(list(X[i]))
          
        for i in range(len(l)):
            dat[i] = np.array(dat[i])
        p = [ i/sum(count[0]) for i in count[0]]
        z = dict(zip(l,p))
        p = []
        
        for i in range(len(l)):
            p.append(self.likelihood(dat[i],t)*z[l[i]])
            
        return l[l.index(p.index(max(p)))]
    
    #predict on the test data 
    def predict(self, testX):
        l = []
        for i in range(len(testX)):
            l.append(self.train(self.X,self.y,testX[i]))
        return l

#instantiation and prediction on our classifier    
clf = Naive_Bayes()
clf.fit(trainX,trainy)
pred = clf.predict(testX)
print(pred)
print(testy)
print(accuracy_score(testy,pred))    