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

# Standardisation des données : chaque aura une moyenne de 0 et un écart-type de 1
# nécessaire pour la régression logistique
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

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

        # On répète la descente de gradient x fois
        for _ in range(self.num_iterations):
            self.update_weights()

    def update_weights(self):

        #TO DO

        # Calcul de z = X*w + b
        z = np.dot(self.X, self.weights) + self.bias

        # Probabilités prédites avec la sigmoide
        y_pred = self.sigmoid(z)

        # Calcul des gradients par rapport aux poids et au biais et / par m
        dw = (1 / self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1 / self.m) * np.sum(y_pred - self.y)

        # Mise à jour des poids et du biais
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db


    def predict(self, X):
        #TO DO

        # Calcul de z puis application de la sigmoide pour avoir les probabilités
        z = np.dot(X, self.weights) + self.bias
        y_prob = self.sigmoid(z)

        # Si la proba est supérieure à 0.5 on prédit 1, sinon 0
        Y = [1 if p > 0.5 else 0 for p in y_prob]
        return np.array(Y)




#TO DO : TEST THE REGRESSION LOGISTIC AND COMPARISION WITH LDA

# On reprend la classe LDA codée dans la partie I
class LDA:
    def __init__(self, param=1e-6):
        self.param = param

    def fit(self, X, y):
        y = np.array(y)
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.means = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))
        self.cov = np.zeros((n_features, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx, :] = X_c.mean(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]
            # Calcul de la matrice de covariance intra-classe
            self.cov += np.dot((X_c - self.means[idx]).T, (X_c - self.means[idx]))

        self.cov /= X.shape[0]
        # Ajout du terme de régularisation sur la diagonale
        self.cov += self.param * np.eye(n_features)
        self.cov_inv = np.linalg.inv(self.cov)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        x = np.array(x).reshape(-1)

        for idx, c in enumerate(self.classes):
            diff = x - self.means[idx]
            # Calcul du score pour chaque classe
            posterior = -0.5 * diff @ self.cov_inv @ diff + np.log(self.priors[idx])
            posteriors.append(posterior)

        # On retourne la classe avec le score le plus élevé
        return self.classes[np.argmax(posteriors)]


# Q11 : test avec lr=0.01 et 1000 itérations
print("\n TEST DE LA REGRESSION LOGISTIQUE (lr=0.01, iter=1000)")
n_runs = 5
acc_lr = []
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    lr = LogisticRegressionCustom(learning_rate=0.01, num_iterations=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred_lr)
    acc_lr.append(acc)
    print(f"  Run {i+1}: {acc*100:.2f}%")
print(f"  Moyenne Logistic Regression: {np.mean(acc_lr)*100:.2f}%")
# Performance Régression Logistique : Excellent pour la classification binaire, converge bien avec la descente de gradient.
# Ses avantages : probabilités interprétables, pas d'hypothèse sur la distribution des données, robuste.
# Ses inconvénients : sensible aux outliers, nécessite standardisation, peut converger lentement.

# Q12 : variation du nombre d'itérations et comparaison avec LDA
print("\n COMPARAISON LR vs LDA EN FAISANT VARIER LE NOMBRE D'ITERATIONS")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lda = LDA()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
acc_lda = accuracy_score(y_test, y_pred_lda)
print(f"  Accuracy LDA : {acc_lda*100:.2f}%")

iterations_list = [100, 500, 1000, 5000, 10000, 20000]
for n_iter in iterations_list:
    lr = LogisticRegressionCustom(learning_rate=0.01, num_iterations=n_iter)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  iter={n_iter} -> LR: {acc*100:.2f}%  |  LDA: {acc_lda*100:.2f}%")

# Analyse des résultats de comparaison LR vs LDA :
# - LDA obtient directement sa performance optimale car c'est une méthode analytique (pas d'itérations)
# - La régression logistique améliore ses performances avec plus d'itérations jusqu'à convergence
# - Avec suffisamment d'itérations, LR peut égaler ou dépasser LDA sur ce dataset
# - LR est plus flexible mais nécessite un réglage du learning rate et du nombre d'itérations
# - Le choix dépend du contexte : LDA pour la rapidité, LR pour la flexibilité et l'interprétabilité
