import joblib
import numpy as np

# ==========================================
# Load Trained Model
# ==========================================
model = joblib.load("models/crop_model.pkl")

# ==========================================
# Prediction Function
# ==========================================
def predict_crop(n, p, k, temperature, humidity, ph, rainfall):

    features = np.array([
        [n, p, k, temperature, humidity, ph, rainfall]
    ])

    prediction = model.predict(features)

    return prediction[0]

# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    print("\n========== Crop Recommendation ==========\n")

    n = float(input("Nitrogen (N): "))
    p = float(input("Phosphorus (P): "))
    k = float(input("Potassium (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("Soil pH: "))
    rainfall = float(input("Rainfall (mm): "))

    crop = predict_crop(
        n,
        p,
        k,
        temperature,
        humidity,
        ph,
        rainfall
    )

    print("\n========================================")
    print("Recommended Crop :", crop)
    print("========================================")