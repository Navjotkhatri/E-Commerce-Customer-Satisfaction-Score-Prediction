import pandas as pd
import numpy as np
import re
import joblib
from nltk.corpus import stopwords

# Load Saved Objects
scaler = joblib.load("models/scaler.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
feature_columns = scaler.feature_names_in_.tolist()

stop_words = set(stopwords.words("english"))


def clean_text(text):

    if pd.isna(text):
        text = "No Remarks"

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)


def preprocess(df):

    # -----------------------------
    # Fill Missing Remarks
    # -----------------------------
    df['Customer Remarks'] = df['Customer Remarks'].fillna("No Remarks")

    # -----------------------------
    # Date Conversion
    # -----------------------------
    date_cols = [
        'Issue_reported at',
        'issue_responded',
        'Survey_response_Date'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            errors='coerce',
            dayfirst=True
        )

    # -----------------------------
    # Response Time
    # -----------------------------
    df['Response_Time_Minutes'] = (
        df['issue_responded'] -
        df['Issue_reported at']
    ).dt.total_seconds() / 60

    df['Response_Time_Minutes'] = df['Response_Time_Minutes'].clip(lower=0)

    # -----------------------------
    # Date Features
    # -----------------------------
    df['Report_Hour'] = df['Issue_reported at'].dt.hour

    df['Report_Day'] = df['Issue_reported at'].dt.day

    df['Report_Month'] = df['Issue_reported at'].dt.month

    df['Report_Weekday'] = df['Issue_reported at'].dt.day_name()

    df['Response_Hour'] = df['issue_responded'].dt.hour

    # -----------------------------
    # Drop Columns
    # -----------------------------
    drop_cols = [
        'Unique id',
        'Order_id',
        'order_date_time',
        'Customer_City',
        'Product_category',
        'Item_price',
        'connected_handling_time',
        'Issue_reported at',
        'issue_responded',
        'Survey_response_Date'
    ]

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # -----------------------------
    # One Hot Encoding
    # -----------------------------
    low_cardinality = [
        'channel_name',
        'category',
        'Sub-category',
        'Tenure Bucket',
        'Agent Shift',
        'Report_Weekday'
    ]

    df = pd.get_dummies(
        df,
        columns=low_cardinality,
        drop_first=True,
        dtype=int
    )

    # -----------------------------
    # TF-IDF
    # -----------------------------
    df['Customer Remarks'] = df['Customer Remarks'].apply(clean_text)

    tfidf_features = tfidf.transform(df['Customer Remarks'])

    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=tfidf.get_feature_names_out()
    )

    df.drop(columns=['Customer Remarks'], inplace=True)

    df = pd.concat(
        [df.reset_index(drop=True),
         tfidf_df.reset_index(drop=True)],
        axis=1
    )

    # -----------------------------
    # Match Training Columns
    # -----------------------------
    for col in feature_columns:

        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # -----------------------------
    # Scaling
    # -----------------------------
    df_scaled = scaler.transform(df)

    return df_scaled