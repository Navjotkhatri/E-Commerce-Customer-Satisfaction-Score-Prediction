from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("models/xgboost_model.json")

print("Model Loaded Successfully")