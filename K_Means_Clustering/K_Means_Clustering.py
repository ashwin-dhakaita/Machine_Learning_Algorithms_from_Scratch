# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:53:58 2018

@author: Ashwin Dhakaita
"""
#import the required modules
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import time

#create datasets
X,y = make_blobs(10,2,2)
print(X)
print(y)

#k-means-clustering class
class K_Means:
    
    #constructor function of our class which takes number of clusters and value of k as input
    def __init__(self, n_clusters, n_runs):
        self.n_clusters = n_clusters
        self.n_runs = n_runs
    
    #computes euclidean distance measure 
    def euclidean_distance(self,x1,x2):
            x1 = np.array(x1)
            x2 = np.array(x2)
            return sum(map(lambda x: x**2,(x2-x1)))**0.5
    
    #fits various datapoints into different clusters
    def fit(self, X):
        centroids =  [X[f] for f in np.random.choice(list(range(len(X))),self.n_clusters)]
        while self.n_runs:
            dist = []
            for i in range(len(X)):
                l = []
                for j in range(self.n_clusters):
                    l.append(self.euclidean_distance(X[i],centroids[j]))
                dist.append(l.index(min(l)))
            print(dist)
            time.sleep(1)
            for i in range(len(centroids)):
                s = []
                for j in range(len(dist)):
                    if dist[j]==i:
                        s.append(X[j])
                c = []
                s = np.array(s)
                for j in range(len(X[0])):
                        c.append(s[:,j].mean())
                centroids[i] = c
            self.n_runs = self.n_runs - 1
                                    
        self.centers = centroids
    
    #predict clusters of the given data
    def predict(self, X):
        dist = []
        for i in range(len(X)):
            l = []
            for j in range(self.n_clusters):
                l.append(self.euclidean_distance(X[i],self.centers[j]))
            dist.append(l.index(min(l)))
        return dist

#instantiation and prediction steps of our clusterer
clf = K_Means(2, 5)
clf.fit(X)
print(clf.predict(X))
                
        