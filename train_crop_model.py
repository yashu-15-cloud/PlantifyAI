import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# Load Dataset
# ==========================================

dataset = pd.read_csv("data/Crop_recommendation.csv")

print(dataset.head())

# ==========================================
# Features & Labels
# ==========================================

X = dataset.drop("label", axis=1)
y = dataset["label"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining Crop Recommendation Model...\n")

model.fit(X_train, y_train)

# ==========================================
# Accuracy
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy*100:.2f}%")

# ==========================================
# Save Model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/crop_model.pkl"
)

print("\n===================================")
print("Training Completed Successfully!")
print(f"Accuracy : {accuracy*100:.2f}%")
print("Model Saved")
print("models/crop_model.pkl")
print("===================================")