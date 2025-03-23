import numpy as np
from utils.models.linear_regression import LinearRegression
from utils.data_loader import load_wine_quality, train_test_split_features_target
from utils.metrics import mean_squared_error


df = load_wine_quality("data/winequality-white.csv")


X_train, y_train, X_test, y_test = train_test_split_features_target(
    df, target_column="quality", test_size=0.2, random_seed=42
)

# No need to convert to numpy arrays, since train_test_split already returns numpy arrays
# X_train = X_train.to_numpy()  # Remove this line
# y_train = y_train.to_numpy()  # Remove this line
# X_test = X_test.to_numpy()  # Remove this line
# y_test = y_test.to_numpy()  # Remove this line

# Train your linear regression model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Linear Regression Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
