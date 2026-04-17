import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([1.2, 2.3, 2.9, 3.8, 5.1, 6.0, 8.0])

model = LinearRegression()
model.fit(X, y)

# Predictions for normal values
X_test = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
y_pred = model.predict(X_test)

# Print predictions
print("Predictions for x =", X_test.flatten())
print("Predicted y =", np.round(y_pred, 2))

# Plot
x_line = np.linspace(0, 22, 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, y, color='orange', label='Data (with outlier)')
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.scatter(X_test, y_pred, color='blue', marker='x', s=100, label='Predictions')
plt.title('Regression with Outlier - Predictions are way off')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
