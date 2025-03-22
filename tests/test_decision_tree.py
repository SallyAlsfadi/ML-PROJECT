from utils.models.decision_tree import DecisionTreeClassifier
from utils.data_loader import load_breast_cancer, train_test_split_features_target
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

df = load_breast_cancer()
X_train, y_train, X_test, y_test = train_test_split_features_target(df, "diagnosis", test_size=0.2, random_seed=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train.to_numpy(), y_train.to_numpy())
y_pred = model.predict(X_test.to_numpy())

print("Decision Tree Evaluation:")


acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print("F1 Score:", round(f1, 4))
print("Confusion Matrix:\n", cm)