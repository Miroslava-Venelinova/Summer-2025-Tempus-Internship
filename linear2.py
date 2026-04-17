import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
data = pd.read_csv(url)

X = data['age'].values.reshape(-1, 1)
y = data['charges'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for training data
y_pred = model.predict(X)

print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Plot data and regression line
plt.scatter(X, y, alpha=0.5, label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Age')
plt.ylabel('Insurance Charges')
plt.title('1D Linear Regression: Age vs Charges (Insurance Dataset)')
plt.legend()
plt.grid(True)
plt.show()
age = 40
predicted_charge = model.coef_[0] * age + model.intercept_
print(f"Predicted insurance charge for age {age}: {predicted_charge:.2f}")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")


