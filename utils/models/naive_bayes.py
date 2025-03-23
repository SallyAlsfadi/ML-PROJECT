import numpy as np

class NaiveBayesClassifier:
    def __init__(self, alpha=1e-9):
        self.alpha = alpha  # Smoothing parameter to avoid zero variance
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.
        
        Parameters:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples,).
        """
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        # Compute priors based on class distribution
        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0) + self.alpha  # Add smoothing to variance
            self.priors[cls] = (X_c.shape[0] + self.alpha) / (X.shape[0] + len(self.classes) * self.alpha)

    def _gaussian_pdf(self, cls, x):
        """
        Calculate the Gaussian probability density function for a given class and input.
        
        Parameters:
            cls: The class label.
            x (np.ndarray): Input features of shape (n_features,).
        
        Returns:
            np.ndarray: Gaussian PDF values for each feature.
        """
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_instance(self, x):
        """
        Predict the class label for a single instance.
        
        Parameters:
            x (np.ndarray): Input features of shape (n_features,).
        
        Returns:
            int: Predicted class label.
        """
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.priors[cls])
            class_conditional = np.sum(np.log(self._gaussian_pdf(cls, x) + self.alpha))  # Add smoothing
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Predict class labels for multiple instances.
        
        Parameters:
            X (np.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        return np.array([self._predict_instance(x) for x in X])

    def __init__(self, alpha=1e-9):
        self.alpha = alpha  # Smoothing parameter to avoid zero variance
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.
        
        Parameters:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples,).
        """
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0) + self.alpha  # Add smoothing to variance
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    def _gaussian_pdf(self, cls, x):
        """
        Calculate the Gaussian probability density function for a given class and input.
        
        Parameters:
            cls: The class label.
            x (np.ndarray): Input features of shape (n_features,).
        
        Returns:
            np.ndarray: Gaussian PDF values for each feature.
        """
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_instance(self, x):
        """
        Predict the class label for a single instance.
        
        Parameters:
            x (np.ndarray): Input features of shape (n_features,).
        
        Returns:
            int: Predicted class label.
        """
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.priors[cls])
            class_conditional = np.sum(np.log(self._gaussian_pdf(cls, x) + self.alpha))  # Add smoothing
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Predict class labels for multiple instances.
        
        Parameters:
            X (np.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        return np.array([self._predict_instance(x) for x in X])