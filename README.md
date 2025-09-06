# Transaction-Review

An interactive and explainable AI dashboard designed to assist financial analysts in detecting, investigating, and managing anomalous transactions.

[https://transaction-copilot-misbah.streamlit.app](https://transaction-copilot-misbah.streamlit.app)

<img width="1920" height="984" alt="image" src="https://github.com/user-attachments/assets/0957b7f0-db18-4bd4-aee6-0fe2eab17a6a" />

<img width="867" height="909" alt="image" src="https://github.com/user-attachments/assets/7608478c-63cc-4377-a538-eb014eea0b41" />

## The Problem
Financial fraud is a massive problem, and analysts often have to sift through thousands of transactions to find suspicious activity. Standard machine learning models can flag transactions but often act as "black boxes," leaving the analyst wondering why a transaction was flagged. This project tackles that problem by not just identifying anomalies but also explaining the reasoning behind each prediction.

## Key Features
Intelligent Anomaly Detection: Utilizes an unsupervised IsolationForest model to score and rank millions of transactions by risk, even in a highly imbalanced dataset.

Explainable AI (XAI) Pipeline: Implements a novel surrogate model approach (XGBoost) to enable the use of SHAP, providing clear, human-readable explanations and waterfall plots for each flagged transaction.

Interactive Review Queue: A paginated, user-friendly interface built with Streamlit that allows analysts to efficiently work through a prioritized list of high-risk cases.

Case Management System: A complete analyst workflow with action buttons to mark cases as "Safe," "Fraud," or "Manual Review," which dynamically updates the queue.

Automated Reporting: A feature to export lists of confirmed fraud or manually reviewed cases directly to a downloadable Excel file.

## Technology Stack
Language: Python

Machine Learning: Scikit-learn, XGBoost

Explainable AI: SHAP

Data Manipulation: Pandas

Dashboarding: Streamlit

Plotting: Matplotlib

## How It Works
The application uses a two-stage model:

An Isolation Forest model first identifies statistical anomalies in the transaction data.

A surrogate XGBoost model, trained to mimic the Isolation Forest, is then used to generate SHAP explanations, providing insight into the "black box" of the primary model.

This entire engine is wrapped in a Streamlit app that provides a user-friendly interface for an analyst to interact with the model's findings.

## How to Run Locally
Clone the repository:

    git clone https://github.com/bagwanmisbah/Transaction-Review-Copilot.git
    cd your-repo
  
Create and activate a virtual environment:

    
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    Install dependencies:

    pip install -r requirements.txt
    
Run the Streamlit app:

    streamlit run app.py
