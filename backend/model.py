import os
import joblib
import pandas as pd
import numpy as np

# Absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "encoder.pkl"))
features = joblib.load(os.path.join(ARTIFACTS_DIR, "features.pkl"))


CAT_COLS = ["category", "gender", "city", "state", "zip"]

def predict_transaction(data: dict):
    df = pd.DataFrame([data])
    df = df[features]

    df[CAT_COLS] = encoder.transform(df[CAT_COLS])
    df_scaled = scaler.transform(df)

    # Prediction
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # --- EXPLAINABILITY ---
    contributions = df_scaled[0] * model.coef_[0]
    feature_importance = sorted(
        zip(features, contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    explanation = [
        {
            "feature": f,
            "impact": round(float(c), 4)
        }
        for f, c in feature_importance
    ]

    return {
        "is_fraud": int(pred),
        "fraud_probability": round(float(prob), 4),
        "top_factors": explanation
    }



if __name__ == "__main__":
    sample_transaction = {
        "category": "grocery_pos",
        "amt": 120.5,
        "gender": "F",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8000000,
        "unix_time": 1325376018,
        "merch_lat": 40.7306,
        "merch_long": -73.9352
    }

    result = predict_transaction(sample_transaction)
    print(result)
