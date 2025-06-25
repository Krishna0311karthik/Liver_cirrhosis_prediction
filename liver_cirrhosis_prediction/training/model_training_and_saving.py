
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("data/liver.csv")  # Rename your file accordingly

# Step 2: Data Preprocessing
df['Dataset'] = df['Dataset'].replace(2, 0)  # 2 -> 0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df.dropna(inplace=True)

# Step 3: Split features and target
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Step 4: Normalize
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()

# Step 8: Save model and normalizer
pickle.dump(model, open("rf_acc_68.pkl", "wb"))
pickle.dump(normalizer, open("normalizer.pkl", "wb"))
print("✅ Model and Normalizer saved.")
