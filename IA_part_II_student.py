
#---------------------------------PROJECT: PART II------------------------#

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.datasets import load_breast_cancer




data = load_breast_cancer() ;
X = data.data
y = data.target

#TO DO: STANDARDIZE THE DATA

import numpy as np

#LOGISTIC REGRESSION

class LogisticRegressionCustom:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y

        #TO DO

    def update_weights(self):
        
        #TO DO
      

    def predict(self, X):
       #TO DO
        return np.array(Y)




#TO DO : TEST THE REGRESSION LOGISTIC AND COMPARISION WITH LDA




