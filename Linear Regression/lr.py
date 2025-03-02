import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data (e.g., advertising budget -> sales)
np.random.seed(42)
X = 1 + 9 * np.random.rand(100, 1)  # Advertising budget (in thousand $)
y = 20 + 5 * X + np.random.randn(100, 1) * 3  # Sales (in thousand units) with noise

# Split into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Model coefficients
w = model.coef_[0][0]
b = model.intercept_[0]
print(f"Linear Regression Model: w = {w:.4f}, b = {b:.4f}")

# Model evaluation
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print(f"MSE (train): {mse_train:.4f}, MSE (test): {mse_test:.4f}")
print(f"R² (train): {r2_train:.4f}, R² (test): {r2_test:.4f}")

# Visualization with an unusual plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_train, y_train, label="Training data", color='blue', alpha=0.7, edgecolors='black')
ax.scatter(X_test, y_test, label="Testing data", color='orange', alpha=0.7, edgecolors='black')
ax.plot(X, model.predict(X), color='red', lw=2, linestyle='dashed', label="Linear Model")
ax.fill_between(X.flatten(), model.predict(X).flatten() - 5, model.predict(X).flatten() + 5, color='red', alpha=0.2, label="Confidence Interval")
ax.set_xlabel("Advertising Budget (thousand $)")
ax.set_ylabel("Sales (thousand units)")
ax.set_title("Linear Regression: Unusual Visualization")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()