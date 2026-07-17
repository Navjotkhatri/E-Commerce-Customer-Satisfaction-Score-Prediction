# 🛍️ Customer Satisfaction (CSAT) Score Prediction using Deep Learning & Machine Learning

## 📌 Project Overview

Customer Satisfaction (CSAT) is one of the most important performance indicators for any e-commerce business. This project aims to predict customer satisfaction scores based on customer support interactions using Artificial Neural Networks (ANN) and compare its performance with traditional Machine Learning models.

The project follows the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, comparison, and local deployment.

---

## 🎯 Objectives

- Analyze customer support interaction data.
- Perform data cleaning and preprocessing.
- Engineer meaningful features from date, time, and text data.
- Build an Artificial Neural Network (ANN) for multi-class CSAT prediction.
- Compare ANN performance with traditional Machine Learning models.
- Deploy the best-performing model locally for real-time prediction.

---

## 📊 Dataset Information

The dataset contains customer support interactions from an e-commerce platform and includes features such as:

- Channel Name
- Category
- Sub-category
- Customer Remarks
- Order Details
- Issue Reported Time
- Issue Response Time
- Customer City
- Product Category
- Item Price
- Agent Details
- Tenure Bucket
- Agent Shift
- CSAT Score (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Joblib
- VS Code
- Google Colab

---

## 📋 Project Workflow

### 1. Data Collection

- Loaded dataset
- Dataset inspection
- Variable understanding

### 2. Data Wrangling

- Missing value handling
- Duplicate removal
- Datetime conversion
- Feature extraction
- Data consistency checks

### 3. Exploratory Data Analysis

- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Correlation Analysis
- Word Cloud Analysis
- Business Insights

### 4. Feature Engineering

- Response Time calculation
- Hour, Day and Month extraction
- Weekday extraction
- Text preprocessing
- TF-IDF Vectorization
- One-Hot Encoding
- Feature Scaling

### 5. Model Building

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)

---

## 📈 Model Performance

| Model | Accuracy |
|--------|-----------|
| XGBoost | **71.82%** |
| Random Forest | **71.02%** |
| Decision Tree | 38.77% |
| Logistic Regression | 35.53% |
| Artificial Neural Network | 34.94% |

**Best Performing Model:** XGBoost

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Balanced Accuracy

---

## 💡 Key Insights

- Customer remarks contain valuable information for satisfaction prediction.
- Longer response times generally reduce customer satisfaction.
- Order-related and refund-related issues contribute significantly to lower CSAT scores.
- Agent tenure and work shifts influence customer experience.
- The dataset is highly imbalanced, affecting ANN performance.

---

## 🚀 Business Impact

The developed prediction system can help businesses:

- Predict customer satisfaction before survey responses are received.
- Identify dissatisfied customers proactively.
- Improve customer support quality.
- Reduce response time.
- Enhance customer retention.
- Support data-driven decision making.

---

## ⚠️ Challenges Faced

- Large number of missing values.
- Highly imbalanced target classes.
- Sparse textual feedback.
- High-cardinality categorical features.
- ANN underperformed on structured tabular data.

---

## 📚 Learning Outcomes

This project provided practical experience in:

- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Text Mining
- Machine Learning
- Deep Learning
- Model Evaluation
- Hyperparameter Tuning
- Model Deployment

---

## 💻 Local Deployment

The best-performing XGBoost model was saved using Joblib and deployed locally in Visual Studio Code.

Saved files include:

- xgboost_csat_model.pkl
- scaler.pkl
- tfidf_vectorizer.pkl
- feature_columns.pkl

These files are loaded during prediction to ensure consistent preprocessing and accurate CSAT score prediction.

---

## 📁 Project Structure

```
Customer-Satisfaction-Prediction/
│
├── Data/
│   └── csat_dataset.csv
│
├── Notebook/
│   └── CSAT_Prediction.ipynb
│
├── Models/
│   ├── xgboost_csat_model.pkl
│   ├── scaler.pkl
│   ├── tfidf_vectorizer.pkl
│   └── feature_columns.pkl
│
├── Deployment/
│   ├── app.py
│   ├── predict.py
│   └── preprocess.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🏆 Conclusion

This project successfully developed a Customer Satisfaction prediction system using both Deep Learning and Machine Learning techniques. Although the Artificial Neural Network fulfilled the deep learning objective, XGBoost achieved the highest accuracy (**71.82%**) and proved to be the most suitable model for this structured and imbalanced dataset. The project demonstrates the complete end-to-end machine learning pipeline, from data preprocessing and feature engineering to model deployment, while providing actionable insights that can help improve customer service quality and overall business performance.

---

## 👨‍💻 Author

**Navjot Khatri**

Machine Learning | Data Science | Artificial Intelligence
