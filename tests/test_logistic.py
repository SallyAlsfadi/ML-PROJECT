import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models.logistic_regression import LogisticRegression
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from utils.data_loader import load_breast_cancer, train_test_split_features_target


df = load_breast_cancer()
X_train, y_train, X_test, y_test = train_test_split_features_target(df, "diagnosis", 0.2, 42)


model = LogisticRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

y_test = np.array(y_test)
y_pred = np.array(model.predict(X_test))

print("y_test distribution:", np.unique(y_test, return_counts=True))
print("y_pred distribution:", np.unique(y_pred, return_counts=True))

print("Logistic Regression Evaluation:")
print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
print(f"Precision: {precision(y_test, y_pred):.4f}")
print(f"Recall: {recall(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))





