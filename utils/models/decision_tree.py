import numpy as np

class DecisionTreeClassifier:
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  

    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain(self, y, X_column, threshold):
        
        parent_entropy = self._entropy(y)

     
        left_idx = X_column <= threshold
        right_idx = X_column > threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0

      
        n = len(y)
        n_left, n_right = np.sum(left_idx), np.sum(right_idx)
        e_left, e_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

      
        return parent_entropy - child_entropy

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

      
        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        best_gain = -1
        best_feat, best_thresh = None, None

        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature_index
                    best_thresh = threshold

        if best_gain == 0:
            return self.Node(value=self._most_common_label(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)
