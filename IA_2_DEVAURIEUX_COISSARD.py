
#---------------------------------PROJECT: PART II------------------------#

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer


#%% Initialisation des données

data = load_breast_cancer()
X = data.data
y = data.target

std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

# Séparation des données en données d'entrainement et données de test
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, train_size=0.8)

#%% LOGISTIC REGRESSION

class LogisticRegressionCustom:
    def __init__(self, learning_rate, num_iterations):
        """
        Entrées :
            learning_rate : taux d'apprentissage pour l'algorithme de descente de gradient
            num_iterations : nombre de répétitions de l'algorithme de descente de gradient
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Entrées :
            X : Données d'entrainement sous forme de matrice
            y : labels d'entrainement sous forme de vecteur
            
        Calcule les poids optimaux (en fonction de num_iteration) de chaque caractéristique
        
        """
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        
        # Répète l'algorithme de descente de gradient num_iterations fois
        for k in range(self.num_iterations):
            self.update_weights()

    def update_weights(self):
        """
        Utilise la méthode de descente de gradient pour chercher un poids cohérent
        pour chaque caractéristique en fonction des données d'entrainement
        
        """
        
        # Parcourt toutes les tumeurs
        for i in range(self.m):
            
            # Calcul de la probabilité à posteriori que la classe y = 1
            Pi = self.sigmoid((self.bias + sum([self.weights[p]*self.X[i][p] for p in range(self.n)])))
            
            # Mise à jour du biais
            self.bias -= self.learning_rate * (Pi - self.y[i])
            
            # Mise à jour de chaque poids
            for p in range(self.n):
                self.weights[p] -= self.learning_rate * (Pi - self.y[i]) * self.X[i][p]
            
    def predict(self, X):
        """
        Entrée :
            X : Données à classer sous forme de matrice (vecteur de vecteurs)
            
        Sortie :
            Y : Vecteur contenant dans l'ordre de X la classe de chaque nouvelle donnée
        """
        
        Y = []
        
        # Parcourt toutes les nouvelles tumeurs
        for i in range(len(X)):
            
            # Calcule la probabilité que la donnée à l'indice i soit de classe y = 1 (donc bénigne)
            Pi = self.sigmoid((self.bias + sum([self.weights[p]*X[i][p] for p in range(self.n)])))
            
            # Ajout dans le vecteur Y de la classe estimée en fonction du résultat de Pi
            if Pi > 0.5 :
                Y.append(1)
            else :
                Y.append(0)
        
        return np.array(Y)

LGR = LogisticRegressionCustom(0.01,100)
LGR.fit(X_train,y_train)
result = LGR.predict(X_test)
precision = accuracy_score(y_test, result)

import matplotlib.pyplot as plt

train_sizes = np.arange(0.1, 1.0, 0.1)
accuracies = []

for train_size in train_sizes:
    
    # Split avec taille variable
    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, train_size=train_size, test_size=1-train_size, random_state=42
    )
    
    # Nouveau modèle à chaque fois
    model = LogisticRegressionCustom(learning_rate=0.01, num_iterations=2000)
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

#%%
# Plot
plt.figure(figsize=(8,5))
bars = plt.bar(train_sizes * 100, accuracies, width=5)
plt.bar_label(bars, fmt='%.2f%%')
plt.xlabel("Pourcentage de données d'entraînement (%)")
plt.ylabel("Accuracy")
plt.title("Performance du modèle en fonction de la taille du training set")
plt.show()


