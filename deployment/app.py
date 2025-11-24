# -----------------------------------------------------------
# SentimentSense - Flask API
# Author: Rajit R Krishna
# Dubai, UAE
#
# Endpoints:
#   /predict  → Predict sentiment (TF-IDF or RoBERTa)
#   /health   → Health check
#   /retrain  → Retrain classical ML model
#
# Project: End-to-End NLP & Deep Learning Sentiment Analysis
# -----------------------------------------------------------

import os
import json
import torch
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------
# Initialize Flask App
# -----------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------
# Load Classical ML Models (TF-IDF + Logistic Regression)
# -----------------------------------------------------------
try:
    tfidf_vectorizer = joblib.load("../src/models/tfidf_vectorizer.pkl")
    logistic_model = joblib.load("../src/models/logistic_regression.pkl")
    print("TF-IDF & Logistic Regression loaded successfully.")
except:
    tfidf_vectorizer = None
    logistic_model = None
    print("Classical models NOT found. Only RoBERTa will work.")

# -----------------------------------------------------------
# Load RoBERTa Model (DistilRoBERTa trained locally)
# -----------------------------------------------------------
try:
    roberta_tokenizer = AutoTokenizer.from_pretrained("../src/models/roberta_model")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("../src/models/roberta_model")
    roberta_model.eval()
    print("RoBERTa model loaded successfully.")
except:
    roberta_tokenizer = None
    roberta_model = None
    print("RoBERTa model NOT found.")

# -----------------------------------------------------------
# Preprocessing function (matches training pipeline)
# -----------------------------------------------------------
def clean_text(text):
    """
    Basic preprocessing before prediction.
    Matches your notebook cleaning steps.
    """
    text = text.lower().strip()
    return text

# -----------------------------------------------------------
# Prediction using Classical TF-IDF + LR
# -----------------------------------------------------------
def predict_tfidf(text):
    cleaned = clean_text(text)
    X = tfidf_vectorizer.transform([cleaned])
    pred = logistic_model.predict(X)[0]

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred]

# -----------------------------------------------------------
# Prediction using RoBERTa
# -----------------------------------------------------------
def predict_roberta(text):
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta_model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred]

# -----------------------------------------------------------
# HEALTH CHECK ENDPOINT
# -----------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200

# -----------------------------------------------------------
# PREDICT ENDPOINT
# -----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    model_to_use = data.get("model", "roberta").lower()

    if model_to_use == "tfidf":
        if logistic_model is None:
            return jsonify({"error": "TF-IDF model not available"}), 500
        sentiment = predict_tfidf(text)

    else:  # Default RoBERTa
        if roberta_model is None:
            return jsonify({"error": "RoBERTa model not available"}), 500
        sentiment = predict_roberta(text)

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "model_used": model_to_use
    })

# -----------------------------------------------------------
# RETRAIN ENDPOINT (ONLY FOR TF-IDF + LR)
# -----------------------------------------------------------
@app.route("/retrain", methods=["POST"])
def retrain():
    """
    This retrains ONLY the classical TF-IDF + Logistic Regression model.
    Roberta retraining must be done manually due to GPU requirement.
    """
    try:
        df = pd.read_csv("../data/synthetic_ecommerce_reviews.csv")

        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        X = df["ReviewText"]
        y = df["Sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tfidf = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_tfidf, y_train)

        joblib.dump(tfidf, "../src/models/tfidf_vectorizer.pkl")
        joblib.dump(clf, "../src/models/logistic_regression.pkl")

        return jsonify({"status": "success", "message": "Model retrained successfully."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
