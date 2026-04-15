
#---------------------------------PROJECT: PART II------------------------#

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.datasets import load_breast_cancer




data = load_breast_cancer();
X = data.data
y = data.target

#STANDARDIZE THE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
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

        for _ in range(self.num_iterations): #integration of the function update_weights
            self.update_weights() 

    def update_weights(self):
        
        z = np.dot(self.X, self.weights) + self.bias #implementation of z
        y_pred = self.sigmoid(z) #Utilisation of sigmoid like said in the report 
        #implementation of the partial derivative 
        #we use the matrix product to get the results more easily
        #if we use the correct formula of J(theta) we use the 1/self.m to normalize the function
        dw = 1/self.m * np.dot(self.X.T, (y_pred - self.y)) #Using .T for X to obtain the desired matrix length
                                                           #partial derivative of weights
        db = 1/self.m * np.sum(y_pred - self.y) # partial derivative of biases
        #Utilisation of the ninth equation
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias #implementation of z
        y_pred = self.sigmoid(z) #utilisation of sigmoid like said in the report
        Y = [1 if i > 0.5 else 0 for i in y_pred] #implementation of the twelveth equation and its parameter to obtain the Y class
        return np.array(Y)

#implementation of the LDA method done in the previous code
class LDA:
    def __init__(self, param = 1e-6):
        self.param = param
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        self.means = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))
        self.cov = np.zeros((n_features, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            self.priors[idx] = X_c.shape[0] / X.shape[0]
            
            self.means[idx, :] = np.mean(X_c, axis=0)
            
            X_c_centered = X_c - self.means[idx, :]
            self.cov += np.dot(X_c_centered.T, X_c_centered)
            
        self.cov /= (X.shape[0]- len(self.classes))
            
        self.cov += np.eye(n_features) * self.param
        self.inv_cov=np.linalg.inv(self.cov)
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            
            mean=self.means[idx]
            prior=self.priors[idx]
                       
            posterior=-0.5*(x-mean).T@self.inv_cov@(x-mean)+np.log(prior)
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

            



#TEST THE REGRESSION LOGISTIC AND COMPARISION WITH LDA
log_reg = LogisticRegressionCustom(learning_rate=0.01, num_iterations=20000)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)

print("--- Logistic regression ---")
print("Accuracy : ", accuracy_score(y_test, predictions))


methodLDA=LDA()
methodLDA.fit(X_train, y_train)
resultatLDA=methodLDA.predict(X_test)
print("--- LDA ---")
print("Accuracy : ", accuracy_score(y_test, resultatLDA))

