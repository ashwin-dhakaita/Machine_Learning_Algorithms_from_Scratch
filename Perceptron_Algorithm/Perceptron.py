#import numpy for mathematical processing in python
import numpy as np  

#create the an array containing different 
#datapoints and the last column of which represents the label whether 0 or 1
x = np.array([[1, 0, 0, 0, 0],
       [1, 0, 0, 1, 1],
       [1, 0, 1, 0, 1],
       [1, 0, 1, 1, 1],
       [1, 1, 0, 0, 1],
       [1, 1, 0, 1, 1],
       [1, 1, 1, 0, 1],
       [1, 1, 1, 1, 1]])

#initial weight vector
w = np.array([0,0,-1,2])

#the loop executes till convergence
conv = False
while not conv:
    flag = 0
    for i in x:
            #perceptron's update conditions
            if i[-1]==1 and np.dot(i[0:-1],w)<0:
                    w = w+i[:-1]
                    flag = 1
            if i[-1]==0 and np.dot(i[0:-1],w)>=0:
                    w = w-i[:-1]
                    flag = 1
    else:
        if flag == 0: conv=True
else:
    #final weights are being printed on convergence
    print(w)