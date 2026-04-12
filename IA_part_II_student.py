
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
# Standardisation des données : on centre (moyenne = 0) et on réduit (écart-type = 1)
# afin que toutes les features aient le même poids lors de la descente de gradient
scaler = StandardScaler()
X = scaler.fit_transform(X)

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
        # Descente de gradient : on met à jour les poids num_iterations fois
        for _ in range(self.num_iterations):
            self.update_weights()

    def update_weights(self):

        #TO DO
        # Calcul des prédictions actuelles : y_hat = sigmoid(X * w + b)
        y_hat = self.sigmoid(np.dot(self.X, self.weights) + self.bias)

        # Calcul du gradient de la loss (cross-entropie) par rapport aux poids
        # dw = (1/m) * X^T * (y_hat - y)
        dw = (1 / self.m) * np.dot(self.X.T, (y_hat - self.y))
        # Calcul du gradient par rapport au biais
        db = (1 / self.m) * np.sum(y_hat - self.y)

        # Mise à jour des poids et du biais par gradient descendant
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
       #TO DO
        # Calcul de la combinaison linéaire pondérée puis passage dans la sigmoïde
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        # Seuillage à 0.5 : probabilité >= 0.5 => classe 1 (maligne), sinon classe 0 (bénigne)
        Y = [1 if p >= 0.5 else 0 for p in y_hat]
        return np.array(Y)




#TO DO : TEST THE REGRESSION LOGISTIC AND COMPARISION WITH LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

n_runs = 5

print("\n TEST DE L'ALGORITHME REGRESSION LOGISTIQUE")
acc_lr = []
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = LogisticRegressionCustom(learning_rate=0.1, num_iterations=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred_lr)
    acc_lr.append(acc)
    print(f"  Run {i+1}: {acc*100:.2f}%")
print(f"  Moyenne Régression Logistique: {np.mean(acc_lr)*100:.2f}%")

print("\n TEST DE L'ALGORITHME LDA")
acc_lda = []
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred_lda)
    acc_lda.append(acc)
    print(f"  Run {i+1}: {acc*100:.2f}%")
print(f"  Moyenne LDA: {np.mean(acc_lda)*100:.2f}%")

print("\nCOMPARAISON DES ALGORITHMES (moyenne sur {} runs)".format(n_runs))
print(f"Régression Logistique : {np.mean(acc_lr)*100:.2f}%")
print(f"LDA                   : {np.mean(acc_lda)*100:.2f}%")
