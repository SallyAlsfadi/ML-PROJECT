import numpy as np
from utils.models.logistic_regression import LogisticRegression

class LogisticRegressionOvR:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}

        for cls in self.classes:
           
            binary_y = np.where(y == cls, 1, 0)

       
            model = LogisticRegression(lr=self.lr, n_iters=self.n_iters)
            model.fit(X, binary_y)

          
            self.models[cls] = model

    def predict(self, X):
        X = np.array(X)

        
        probs = {}
        for cls, model in self.models.items():
            linear_output = np.dot(X, model.weights) + model.bias
            class_probs = model.sigmoid(linear_output)
            probs[cls] = class_probs

       
        all_probs = np.array([probs[cls] for cls in self.classes]) 
        preds = np.argmax(all_probs, axis=0)
        return self.classes[preds]
