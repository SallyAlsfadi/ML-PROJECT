from utils.models.logistic_regression import LogisticRegression
from utils.models.k_nearest_neighbors import KNearestNeighbors
from utils.models.naive_bayes import NaiveBayesClassifier
from utils.models.decision_tree import DecisionTreeClassifier
from utils.data_loader import load_mushroom, train_test_split_features_target
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

import numpy as np


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    print(f"{model_name} Evaluation:")
    print("Accuracy:", round(accuracy(y_test, y_pred), 4))
    print("Precision:", round(precision(y_test, y_pred), 4))
    print("Recall:", round(recall(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



mushroom_df = load_mushroom()
X_train, y_train, X_test, y_test = train_test_split_features_target(
    mushroom_df, target_column="class", test_size=0.2, random_seed=42
)


evaluate_model(LogisticRegression(lr=0.01, n_iters=1000), X_train, y_train, X_test, y_test, "Logistic Regression")
evaluate_model(DecisionTreeClassifier(max_depth=10), X_train, y_train, X_test, y_test, "Decision Tree")
evaluate_model(KNearestNeighbors(k=3), X_train, y_train, X_test, y_test, "K-Nearest Neighbors")
evaluate_model(NaiveBayesClassifier(), X_train, y_train, X_test, y_test, "Naive Bayes")
