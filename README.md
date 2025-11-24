# SentimentSense – End-to-End NLP & Deep Learning Sentiment Analysis for E-Commerce Reviews

**Author:** Rajit R Krishna | Data Scientist | 
**Domain:** E-Commerce | NLP | Deep Learning | MLOps

---

## 🔹 Project Overview

E-commerce platforms receive **thousands of customer reviews daily**, but extracting actionable insights manually is impossible.  
**SentimentSense** automates this process by providing:

- Accurate **sentiment classification** of reviews  
- Real-time **prediction API**  
- Support for **noisy and slang text**  
- **Explainable NLP workflow**  
- Structured outputs ready for dashboards  

This project leverages **classic NLP techniques** and **transformer-based models** (RoBERTa) to deliver **robust, production-ready sentiment analysis**.

---

## 🔹 Tech Stack

- **Python 3.x**  
- **NLTK, VADER** (baseline sentiment analysis)  
- **Hugging Face Transformers (RoBERTa)**  
- **Scikit-learn** (Logistic Regression, Random Forest)  
- **Flask API** (deployment)  
- **Pandas, NumPy, Matplotlib, Seaborn** (EDA & visualization)  
- **Git & GitHub** (version control)  

---

## 🔹 Key Features

- **Synthetic UAE-ready dataset** generation for e-commerce reviews  
- **Exploratory Data Analysis (EDA)** with wordclouds, n-grams, and sentiment patterns  
- **Preprocessing pipeline**: tokenization, lemmatization, stopword & slang removal, TF-IDF, RoBERTa tokenization  
- **Modeling**: VADER, Logistic Regression, Random Forest, RoBERTa fine-tuning  
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC  
- **Deployment**: Flask API with `/predict`, `/health`, `/retrain` endpoints  

---

## 🔹 Project Structure

/data # Datasets (synthetic & raw)
/notebooks/eda # Exploratory Data Analysis
/notebooks/preprocessing # Data cleaning & feature engineering
/notebooks/modeling # Model training & evaluation
/src # Core scripts & utilities
/deployment # Flask API code
/pipeline # Automation scripts (optional Airflow DAG)
/docs # Screenshots, diagrams, documentation
