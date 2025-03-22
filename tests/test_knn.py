from utils.models.k_nearest_neighbors import KNearestNeighbors
from utils.data_loader import load_breast_cancer, train_test_split_features_target
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
import numpy as np


df = load_breast_cancer()
X_train, y_train, X_test, y_test = train_test_split_features_target(df, target_column="diagnosis", test_size=0.2, random_seed=42)


model = KNearestNeighbors(k=5)
model.fit(X_train.values, y_train.values)


y_pred = model.predict(X_test.values)
y_pred = np.array(y_pred) 
print("y_pred:", y_pred)
print("y_test:", y_test.values)

print("K-Nearest Neighbors Evaluation:")
print(f"Accuracy: {accuracy(y_test.values, y_pred):.4f}")
print(f"Precision: {precision(y_test.values, y_pred):.4f}")
print(f"Recall: {recall(y_test.values, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test.values, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test.values, y_pred))
