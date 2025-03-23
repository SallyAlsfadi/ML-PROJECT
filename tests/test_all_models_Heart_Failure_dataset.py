from utils.models.logistic_regression import LogisticRegression
from utils.models.k_nearest_neighbors import KNearestNeighbors
from utils.models.naive_bayes import NaiveBayesClassifier
from utils.models.decision_tree import DecisionTreeClassifier
from utils.data_loader import load_heart_failure_data, train_test_split_features_target
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision: {precision(y_test, y_pred):.4f}")
    print(f"Recall: {recall(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Load and prepare heart failure dataset
df = load_heart_failure_data("data/heart_failure_clinical_records_dataset.csv")
X_train, y_train, X_test, y_test = train_test_split_features_target(df, "DEATH_EVENT", test_size=0.2, random_seed=42)

# Evaluate all models
evaluate_model(LogisticRegression(lr=0.05, n_iters=3000), X_train, y_train, X_test, y_test, "Logistic Regression")
evaluate_model(DecisionTreeClassifier(max_depth=10), X_train, y_train, X_test, y_test, "Decision Tree")
evaluate_model(KNearestNeighbors(k=3), X_train, y_train, X_test, y_test, "K-Nearest Neighbors")
evaluate_model(NaiveBayesClassifier(), X_train, y_train, X_test, y_test, "Naive Bayes")
