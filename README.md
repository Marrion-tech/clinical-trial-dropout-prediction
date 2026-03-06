# Clinical Trial Patient Dropout Prediction

## Overview
A machine learning project built during Agilisium Remote Internship to predict 
which clinical trial patients are at risk of dropping out — enabling proactive 
intervention before it happens.

## Problem Statement
Clinical trials suffer from 30-40% dropout rates, costing pharma companies 
billions. This project uses XGBoost to predict dropout risk from patient 
demographics, visit compliance, and trial engagement data.

## Tech Stack
- Python, XGBoost, Scikit-learn, SHAP
- Streamlit Dashboard
- Pandas, NumPy, Matplotlib, Seaborn

## Project Structure
- data/      — Dataset and train/test splits
- notebooks/ — Step by step ML pipeline (4 notebooks)
- models/    — Trained XGBoost model and scaler
- outputs/   — Charts, ROC curve, SHAP plots, risk scores
- app.py     — Interactive 4-page Streamlit dashboard

## How to Run
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn streamlit plotly
streamlit run app.py

## Results
- Model      : XGBoost Classifier (tuned with GridSearchCV)
- Metrics    : Accuracy, F1 Score, AUC-ROC
- Explainability : SHAP waterfall + summary plots
- Dashboard  : 4-page interactive risk monitoring app

## Internship
Agilisium Consulting — Remote Internship
