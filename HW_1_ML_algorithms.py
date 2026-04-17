import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("SAT_GPA.csv")

X = data['SAT'].values.reshape(-1, 1)
Y = data['GPA'].values

model = LinearRegression()
model.fit(X, Y)

y_prediction = model.predict(X)

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intersept: {model.intercept_:.2f}")

plt.scatter(X, Y, alpha=0.5, label='Data points')
plt.plot(X, y_prediction, color='red', label='Regression line')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.title('GPA vs SAT scores')
plt.legend()
plt.grid(True)
plt.show()

sat_score = 1750

predicted_gpa = model.coef_[0] * sat_score + model.intercept_
print(f"Predicted GPA based on SAT score: {sat_score}: {predicted_gpa:.2f}")

mae = mean_absolute_error(Y, y_prediction)
mse = mean_squared_error(Y, y_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(Y, y_prediction)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")
