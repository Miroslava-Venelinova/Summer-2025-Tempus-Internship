

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data
url = "https://gist.githubusercontent.com/trantuyen082001/"
url += "1fc2f5c0ad1507f40e721e6d18b34138/raw/heart.csv"
data = pd.read_csv(url)

# Separate features and target (rename 'output' to 'target' if needed)
X = data.drop('output', axis=1)
y = data['output']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["No Disease","Disease"],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()
