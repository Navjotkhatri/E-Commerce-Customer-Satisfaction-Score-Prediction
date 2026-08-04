import joblib
from preprocess import preprocess

from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("models/xgboost_model.json")


def predict_csat(input_df):
    """
    Predict CSAT Score
    """

    processed_data = preprocess(input_df)

    prediction = model.predict(processed_data)

    # Convert back from 0-4 to 1-5
    return int(prediction[0]) + 1