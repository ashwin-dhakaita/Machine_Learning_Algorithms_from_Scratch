# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:58:07 2018

@author: Ashwin Dhakaita
"""

#import the required modules
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math
import time
from sklearn.metrics import accuracy_score

#import the datasets
data = pd.read_csv("H://zoo.csv",header=None)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X.drop(columns=0,inplace=True)

#split the datasets into training and test sets
trainX, testX, trainy, testy = train_test_split(X,y,test_size=0.2)

#convert numpy series objects into numpy array
trainX = np.array(trainX)
trainy = np.array(trainy)
testX = np.array(testX)
testy = np.array(testy)

#class decision tree
class Decision_Tree:
    
    #intakes data on which the tree is to be built 
    def fit(self, trainX, trainy):
        self.X = trainX
        self.y = trainy
        self.tree = self.ID3(trainX, trainy, trainy)
    
    #computes entropy of the data
    def entropy(self,y):
        val,counts = np.unique(y,return_counts=True)
        s = [-(counts[i]/sum(counts))*(math.log(counts[i]/sum(counts),2)) for i in range(len(val))]
        return sum(s)
    
    #we use information gain measure to search for the node on which splitting occurs
    def information_gain(self,X,feature,y):
        val,counts = np.unique(X[:,feature],return_counts=True)
        total_entropy = self.entropy(y)
        entro = [(counts[i]/sum(counts))*self.entropy(y[X[:,feature]==val[i]]) for i in range(len(val))]
        info_gain = total_entropy - sum(entro)
        return info_gain
    
    #the id3 algorithm for generation of the tree
    def ID3(self,X, y, original_data, parent_class=None):
        if (len(X))==0:
            return parent_class
          
        if (len(X[0,:])==0):
            return parent_class
        
        if (len(np.unique(y))<=1):
            return np.unique(y)
        
        parent_class,c = np.unique(y,return_counts=True)
        parent_class = parent_class[np.argmax(c)]
        info_gain = [self.information_gain(X,i,y) for i in range(1,len(X[0]))]
        best_feature = np.argmax(info_gain)
        tree = {best_feature:{}}
        for i in np.unique(X[:,best_feature]):
            sub_data = np.array([X[j] for j in range(len(X)) if X[j][best_feature]==i])
            sub_y = np.array([y[j] for j in range(len(X)) if X[j][best_feature]==i ])
            sub_data = pd.DataFrame(sub_data)
            sub_data.drop(columns=best_feature,inplace=True)
            sub_data = np.array(sub_data)
            sub_tree = self.ID3(sub_data,sub_y,y,parent_class)
            tree[best_feature][i]=sub_tree
            
        return tree
    
    #predicts classes of the data based on the fitted tree by quering down the tree
    def predict(self, testX):
      l = []
      for i in testX:
           tr = self.tree
           f = list(i)
           flag = 0
           for j in range(len(f)):
               
               try:
                   key = f[list(tr.keys())[0]]
                   tr = tr[list(tr.keys())[0]][key]

               except:
                   if(isinstance(tr,dict)==False):
                       if(isinstance(tr,type(np.array([])))):
                           l.append(tr[0])
                           flag=1
                           break
                       
           if flag==0:
                val,counts = np.unique(testy,return_counts=True)
                l.append(val[np.argmax(counts)])
                  # print(l)
                  # time.sleep(2)
   #   print(l)
      return np.array(l)

#instantiation and prediction on our tree
clf = Decision_Tree()
clf.fit(trainX, trainy)
pred = clf.predict(testX)
print(pred)
print(accuracy_score(testy,pred))