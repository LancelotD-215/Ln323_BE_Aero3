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


# KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
     
        #TO DO



#NAIVEBAYES

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._priors = np.zeros(n_classes)
        self._means = np.zeros((n_classes, n_features))
        self._variances = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self._classes):
           #TO DO
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
         #TO DO
        
        return self._classes[np.argmax(posteriors)]
    

    

#LDA

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
       
            #TO DO
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            
            #TO DO
            
        return self.classes[np.argmax(posteriors)]

            


#TO DO: TEST OF THE ALGORITHMS 






