from utils.models.logistic_regression_ovr import LogisticRegressionOvR
from utils.data_loader import load_robot_failures, train_test_split_features_target
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":

    df = load_robot_failures()
    X_train, y_train, X_test, y_test = train_test_split_features_target(
        df, target_column="failure_type", test_size=0.2, random_seed=42
    )

 
    evaluate_model(LogisticRegressionOvR(lr=0.01, n_iters=3000), X_train, y_train, X_test, y_test, "Logistic Regression (OvR)")
