from utils.data_loader import load_breast_cancer, train_test_split_features_target
from utils.models.logistic_regression import LogisticRegression
from utils.models.decision_tree import DecisionTreeClassifier
from utils.models.k_nearest_neighbors import KNearestNeighbors
from utils.models.naive_bayes import NaiveBayesClassifier
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision: {precision(y_test, y_pred):.4f}")
    print(f"Recall: {recall(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


df = load_breast_cancer()
X_train, y_train, X_test, y_test = train_test_split_features_target(df, "diagnosis", test_size=0.2, random_seed=42)


evaluate_model(LogisticRegression(lr=0.01, n_iters=1000), X_train, y_train, X_test, y_test, "Logistic Regression")
evaluate_model(DecisionTreeClassifier(max_depth=10), X_train, y_train, X_test, y_test, "Decision Tree")
evaluate_model(KNearestNeighbors(k=3), X_train, y_train, X_test, y_test, "K-Nearest Neighbors")
evaluate_model(NaiveBayesClassifier(), X_train, y_train, X_test, y_test, "Naive Bayes")