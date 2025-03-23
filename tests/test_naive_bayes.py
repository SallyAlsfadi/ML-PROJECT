
import numpy as np
from utils.data_loader import load_breast_cancer, train_test_split_features_target
from utils.models.naive_bayes import NaiveBayesClassifier
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

df = load_breast_cancer()
X_train, y_train, X_test, y_test = train_test_split_features_target(df, target_column="diagnosis", test_size=0.2, random_seed=42)


X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

model = NaiveBayesClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Naive Bayes Evaluation:")
print("Accuracy:", round(accuracy(y_test, y_pred), 4))
print("Precision:", round(precision(y_test, y_pred), 4))
print("Recall:", round(recall(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

