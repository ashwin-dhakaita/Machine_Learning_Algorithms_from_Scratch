# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 18:04:34 2018

@author: Ashwin Dhakaita
"""
#import required modules
import pandas as pd
import numpy as np
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#import the dataset
data = pd.read_csv("H://iris.csv",header = None)

#split the data into training and test sets
trainX , testX , trainy , testy = train_test_split(data.iloc[:,:-1],data.iloc[:,-1])

#class K-nearest-neighbours
class K_nearest_neighbours:
    
    def __init__(self, K=None):
        self.K = K
        self.dist_dict = dict()
    def dist (self, x1 , x2):
        return (sum((x2-x1).map(lambda x: x**2)))**0.5
    
    def fit (self, trainX , trainy):
        self.trainX = trainX
        self.trainy = trainy
        
    def predict(self, testX):
        self.distance = [ [ (i,self.dist(v,x)) for i,x in self.trainX.iterrows() ] for j,v in testX.iterrows()]
        self.distance = [ sorted(x,key = lambda x:x[1]) for x in self.distance]
        self.distance = [ x[0:self.K] for x in self.distance]
        print(self.distance[0])
        self.distance = [ [trainy[i] for i,v in j] for j in self.distance ]
        self.distance = [ mode(i) for i in self.distance]
        print(self.distance)
    
#instantiate the class choosing k=7
KNN = K_nearest_neighbours(K=7)
KNN.fit(trainX,trainy)
KNN.predict(testX)

#instantiate the sklearn knn classifier to compare accuracies
KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(trainX,trainy)
print(KNN.predict(testX))

