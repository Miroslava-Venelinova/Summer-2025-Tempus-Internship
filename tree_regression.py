import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("StudentsPerformance.csv")

df = df.dropna()

features_cat = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
features_num = ['reading score', 'writing score']

X_cat = df[features_cat]
X_num = df[features_num]

y = df['math score']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_cat_encoded = encoder.fit_transform(X_cat)

X_full = np.hstack([X_cat_encoded, X_num.values])

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nR² на тестовия набор: {r2:.3f}")

sample_cat = pd.DataFrame({
    'gender': ['female'],
    'race/ethnicity': ['group B'],
    'parental level of education': ['bachelor\'s degree'],
    'lunch': ['standard'],
    'test preparation course': ['none']
})

sample_num = np.array([[80, 78]])

sample_cat_encoded = encoder.transform(sample_cat)

sample_full = np.hstack([sample_cat_encoded, sample_num])

predicted_score = model.predict(sample_full)
print(f"\nПредполагаем резултат по математика: {predicted_score[0]:.2f}")

