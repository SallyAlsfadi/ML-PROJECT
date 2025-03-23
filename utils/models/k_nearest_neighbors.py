import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
      
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
    
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X]).astype(int)

    def _predict_single(self, x):
       
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
