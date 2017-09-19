
# This code is for facial expression recognition using softmax logistic regression.
# We have 8 classes of expressions:
# 1.  0=Angry, 
# 2.  1=Disgust,
# 3.  2=Fear, 
# 4.  3=Happy,
# 5.  4=Sad,
# 6.  5=Surprise,
# 7.  6=Neutral
#  Training set has 28,709 examples. It is a csv file of 48x48 grayscale image pixels, later stored in a 1-dimensional vector.
#  Here we use softmax function as the hypothesis to solve multi-class classification. 


from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

#from softmaxreg_util import getData, softmax, getCost, generate_indicator, getError, getAccuracy
from sklearn.utils import shuffle

# Define the Hypothesis function = softmax function
def softmax(M):
    expM = np.exp(M)
    return expM / expM.sum(axis=1, keepdims=True)

# Cross entropy cost function used in softmax
def getCost(T, Y):
    return -(T*np.log(Y)).sum()

# To get accuracy between prediction and target
def getAccuracy(target, prediction):
    return np.mean(target == prediction)

# To get error between prediction and target
def getError(target, prediction):
    return np.mean(target != prediction)

# To convert a target vetor of labels to a Nx7 indicator matrix of 0's and 1's
def generate_indicator(y):
    N = len(y)
    K = len(set(y))
    indicator = np.zeros((N, K))
    for i in range(N):
        indicator[i, y[i]] = 1
    return indicator

# To get data from csv
# Images are 48x48 = 2304 size vectors, Number of samples, N = 35887
def getData(balance_ones=True):
    
    Y = []
    X = []
    first = True # skip first line in CSV (header)
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0])) # first column - label
            X.append([int(p) for p in row[1].split()]) # second column - space seperated pixels

    # Normalize array X and convert X, Y to numpy arrays        
    X, Y = np.array(X) / 255.0, np.array(Y)

    # Balance the classes
    if balance_ones:
        # balance 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0) # repeating class 1, 9 times
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


class LogisticModel(object):

    # Constructor does nothing
    def __init__(self):
        pass


    def fit(self, X, Y, learning_rate=10e-8, regularization=10e-12, epoch=10000, show_fig=False):
        X, Y = shuffle(X, Y)

        print("Fetching the training and testing set..")

        # split validation/test set
        Xvalid, Yvalid = X[-1000:], Y[-1000:] 
        Tvalid = generate_indicator(Yvalid) # set indicator matrix
        X, Y = X[:-1000], Y[:-1000] # split training data set

        N, D = X.shape
        K = len(set(Y)) # number of classes from Y
        
        print("Initializing the weight matrix and bias vector..")

        self.W = np.random.randn(D, K) / np.sqrt(D) # initiating weight matrix
        self.b = np.zeros(K) # initiating weight vector
        T = generate_indicator(Y) # generate indicator matrix

        min_error = 1
        costs = []  # Initialize a cost function matrix to plot at end
        
        print("Entering the training loop..")

        for i in range(epoch):

            prediction = self.forward(X) # prediction value through softmax forward propogation

            # Gradient descent on softmax function
            grad_W = X.T.dot(prediction - T) + regularization*self.W
            grad_b = (prediction - T).sum(axis=0) + regularization*self.b

            # Update bias and weight
            self.W -= learning_rate*grad_W
            self.b -= learning_rate*grad_b

            if i % 10 == 0:
                predictionvalid = self.forward(Xvalid)
                c = getCost(Tvalid, predictionvalid) # J = T. log(Y)
                costs.append(c)
                a = getAccuracy(Yvalid, np.argmax(predictionvalid, axis=1))
                e = getError(Yvalid, np.argmax(predictionvalid, axis=1))

                print("i:", i, "cost:", c, "Accuracy:", a)
                if e < min_error:
                    min_error = e

        print("min_error:", min_error)

        # Plot cost function vs epoch
        if show_fig:
            plt.plot(costs)
            plt.show()

        # To visualize the weight matrix for class 4
        classToVisulaize = 4
        plt.imshow(scipy.reshape(W[:,classToVisulaize],[28,28]))


    def forward(self, X):
        return softmax(X.dot(self.W) + self.b) 
   

def main():
    X, Y = getData()
    
    model = LogisticModel()
    model.fit(X, Y, show_fig=True)

    

if __name__ == '__main__':
    main()
    
    
