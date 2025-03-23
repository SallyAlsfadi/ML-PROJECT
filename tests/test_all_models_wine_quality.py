from utils.models.logistic_regression import LogisticRegression
from utils.models.k_nearest_neighbors import KNearestNeighbors
from utils.models.naive_bayes import NaiveBayesClassifier
from utils.models.decision_tree import DecisionTreeClassifier
from utils.data_loader import train_test_split_features_target, min_max_scaling
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

import pandas as pd
import numpy as np

def balance_dataset(df, target_column="quality"):

    class_0 = df[df[target_column] == 0]
    class_1 = df[df[target_column] == 1]


    class_0_balanced = class_0.sample(n=len(class_1), random_state=42)
    df_balanced = pd.concat([class_0_balanced, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced

def load_wine_quality(filepath):
    """
    Loads and preprocesses the wine quality dataset (classification version).
    Binarizes quality: 1 if quality >= 7, else 0.
    """
    df = pd.read_csv(filepath, sep=';')
    
    # Binarize target
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

    # Min-Max scale all other features
    for col in df.columns:
        if col != "quality":
            df[col] = min_max_scaling(df[col])

    return df



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
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")


if __name__ == "__main__":
    # Load and balance the data
    df = load_wine_quality("data/winequality-white.csv")
    df = balance_dataset(df, target_column="quality")  

    X_train, y_train, X_test, y_test = train_test_split_features_target(
        df, target_column="quality", test_size=0.2, random_seed=42
    )

    # Evaluate all models
    evaluate_model(LogisticRegression(lr=0.01, n_iters=3000), X_train, y_train, X_test, y_test, "Logistic Regression (Balanced)")
    evaluate_model(DecisionTreeClassifier(max_depth=10), X_train, y_train, X_test, y_test, "Decision Tree")
    evaluate_model(KNearestNeighbors(k=3), X_train, y_train, X_test, y_test, "K-Nearest Neighbors")
    evaluate_model(NaiveBayesClassifier(), X_train, y_train, X_test, y_test, "Naive Bayes")