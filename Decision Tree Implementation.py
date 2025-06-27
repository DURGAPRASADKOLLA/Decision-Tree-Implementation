#step 1:
#install the required libraries 
#pip install scikit-learn pandas matplotlib seaborn

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.datasets import load_breast_cancer
import pandas as pd

#Step 2: Load the Breast Cancer Dataset

breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')  # 0 = malignant, 1 = benign

# Step 3: Explore the Data (Optional)

print(X.head())
print("Target classes:", data.target_names)

#Step 4: Split the Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

#Step 5: Train the Decision Tree Model

from sklearn.tree import DecisionTreeClassifier
# Create model
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
model.fit(X_train, y_train)

#Step 6: Evaluate the Model

from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

#Step 7: Visualize the Decision Tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=data.feature_names, 
          class_names=data.target_names, 
          filled=True, 
          rounded=True)
plt.title("Decision Tree Visualization (Breast Cancer Dataset)")
plt.show()
