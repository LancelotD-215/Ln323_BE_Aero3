#---------------------------------PROJECT: PART I---------------------#

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import zipfile
import requests
from io import BytesIO



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(BytesIO(response.content))

# Extraire le fichier SMSSpamCollection
with zip_file.open('SMSSpamCollection') as file:
    df = pd.read_csv(file, sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.to_csv('SMSSpamCollection_processed.csv', index=False)
X = df['message']
y = df['label']

#TO DO: PRE-PROCESS THE DATA
# Pre-processing

# Initialisation du vectoriseur
vectorizer = CountVectorizer()

# Transformation des messages en matrice de comptage
X_vectorized = vectorizer.fit_transform(X)
# La dimension de chaque vecteur est ainsi le nombre total de mots présent dans le vocabulaire
print(f"Nombre de messages (N) : {X_vectorized.shape[0]}")
print(f"Dimension de l'espace des caractéristiques (P) : {X_vectorized.shape[1]}")

# Séparation des données en données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)
print(f"Taille de y_train : {len(y_train)}")
print(f"Taille de y_test : {len(y_test)}")

# KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        pass
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # converstion des matrices sparse en array pour le calcul de distance
        if hasattr(self.X_train, 'toarray'):
            X_train_array = self.X_train.toarray()
        else:
            X_train_array = self.X_train
        if hasattr(x, 'toarray'):
            x_array = x.toarray().reshape(-1)
        else:
            x_array = x.reshape(-1)
        
        # Calculer les distances entre les données d'entrainement et la donnée de test
        distances = np.linalg.norm(X_train_array - x_array, axis=1)
        # On prend les k les plus proches
        k_indices = np.argsort(distances)[:self.k] # k_indices est un tableau de taille k contenant les indices dans X_train des k plus proches
        k_classe_prox = self.y_train.iloc[k_indices] # Utilisation de .iloc pour éviter les problèmes d'index pandas
        # On retourne la classe majoritaire parmi les voisins les plus proches
        majoritaire = np.bincount(k_classe_prox).argmax() # np.bincount retourne un tableau des apparitions de chaque classe dans k_classe_prox, argmax retourne l'indice de la classe qui a le plus d'apparitions
        return majoritaire


#NAIVEBAYES

class NaiveBayes:
    def fit(self, X, y):
        y = np.array(y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._priors = np.zeros(n_classes)
        self._means = np.zeros((n_classes, n_features))
        self._variances = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            if hasattr(X_c, 'toarray'):
                X_c = X_c.toarray()
            self._priors[idx] = X_c.shape[0] / n_samples
            self._means[idx, :] = X_c.mean(axis=0)
            self._variances[idx, :] = X_c.var(axis=0)
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        if hasattr(x, 'toarray'):
            x = x.toarray().reshape(-1)
        else:
            x = np.array(x).reshape(-1)
        
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            # On ajoute un petit epsilon pour éviter log(0)
            var = self._variances[idx] + 1e-9
            posterior = -0.5 * np.sum(((x - self._means[idx]) ** 2) / var + np.log(var))
            posterior += prior
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
        



#LDA

class LDA:
    def __init__(self, param = 1e-6):
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
            if hasattr(X_c, 'toarray'):
                X_c = X_c.toarray()
            self.means[idx, :] = X_c.mean(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]
            self.cov += np.dot((X_c - self.means[idx]).T, (X_c - self.means[idx]))

        self.cov /= X.shape[0]
        self.cov += self.param * np.eye(n_features)
        self.cov_inv = np.linalg.inv(self.cov)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        if hasattr(x, 'toarray'):
            x = x.toarray().reshape(-1)
        else:
            x = np.array(x).reshape(-1)
        
        for idx, c in enumerate(self.classes):
            diff = x - self.means[idx]
            posterior = -0.5 * diff @ self.cov_inv @ diff + np.log(self.priors[idx])
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

            


#TO DO: TEST OF THE ALGORITHMS 

n_runs = 5

print("\n TEST DE L'ALGORITHME KNN")
acc_knn = []
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred_knn)  # Plus il y a de données d'entrainement plus l'accuracy-score est élevé
    # Ce résultat est logique car plus on a de données d'entrainement plus l'algorithme sait voir les types de messages. 
    # Cependant si on a trop de données d'entrainement on a moins de données de test donc moins de précision
    acc_knn.append(acc)
    print(f"  Run {i+1}: {acc*100:.2f}%")
print(f"  Moyenne KNN: {np.mean(acc_knn)*100:.2f}%")
# Performance KNN : L'algorithme KNN est simple et efficace pour ce type de classification. 
# Ses avantages : pas de phase d'apprentissage, fonctionne bien avec peu de données.
# Ses inconvénients : sensible au dimensionnement élevé, temps de prédiction long sur gros datasets.

print("\n TEST DE L'ALGORITHME NAIVE BAYES")
acc_nb = []
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred_nb)
    acc_nb.append(acc)
    print(f"  Run {i+1}: {acc*100:.2f}%")
print(f"  Moyenne Naive Bayes: {np.mean(acc_nb)*100:.2f}%")
# Performance Naive Bayes : Excellent pour la classification de texte grâce à l'hypothèse d'indépendance des mots.
# Ses avantages : rapide à entraîner, fonctionne bien avec peu de données, résistant au sur-apprentissage.
# Ses inconvénients : hypothèse d'indépendance parfois irréaliste, sensible aux features corrélées.

print("\n TEST DE L'ALGORITHME LDA")
acc_lda = []
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred_lda)
    acc_lda.append(acc)
    print(f"  Run {i+1}: {acc*100:.2f}%")
print(f"  Moyenne LDA: {np.mean(acc_lda)*100:.2f}%")
# Performance LDA : Méthode linéaire robuste qui assume une distribution gaussienne des données.
# Ses avantages : réduit la dimensionnalité, gère bien les données gaussiennes, stable numériquement.
# Ses inconvénients : assume des covariances égales entre classes, peut mal performer si hypothèses incorrectes.

print("\nCOMPARAISON DES ALGORITHMES (moyenne sur {} runs)".format(n_runs))
print(f"KNN (k=3)    : {np.mean(acc_knn)*100:.2f}%")
print(f"Naive Bayes  : {np.mean(acc_nb)*100:.2f}%")
print(f"LDA          : {np.mean(acc_lda)*100:.2f}%")