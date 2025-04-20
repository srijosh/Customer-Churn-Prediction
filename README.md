# Customer Churn Prediction

This project focuses on building a machine learning model to predict customer churn in the telecom sector using the Telco Customer Churn dataset. It utilizes various classification algorithms, handles class imbalance with SMOTE, and offers insights through a Streamlit dashboard.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Tools and Technologies](#tools-and-technologies)

## Introduction

Customer churn is one of the key challenges for service providers. This project builds a robust pipeline using preprocessing, feature engineering, model training, and hyperparameter tuning to predict whether a customer will leave the company.

## Introduction

- Dataset Source: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Features

- Exploratory Data Analysis (EDA) with matplotlib and seaborn

- Label Encoding for categorical variables

- SMOTE for handling class imbalance

- Model comparison (Decision Tree, Random Forest, XGBoost)

- Hyperparameter Tuning with RandomizedSearchCV

- Evaluation Metrics: Accuracy, F1-score, ROC-AUC, Confusion Matrix

- Interactive UI built with Streamlit

## Installation

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Customer-Churn-Prediction.git
```

2. Navigate to the project directory:

```
   cd Customer-Churn-Prediction
```

3. Install the required dependencies:

```
   pip install -r requirements.txt
```

4. Set up environment variables by creating a .env file (Use .env.sample as a reference for setting up your .env file.)

## Usage

To launch the Streamlit dashboard:

```
streamlit run app.py
```

## Modeling

Three models were trained and compared:

- Decision Tree

- Random Forest

- XGBoost

- 📌 Random Forest outperformed others with the highest accuracy using default parameters. Then Hyperparameter Tuning was performed.

## Tools and Technologies

- Python

- Pandas, NumPy

- Matplotlib, Seaborn

- Scikit-learn

- Imbalanced-learn (SMOTE)

- XGBoost

- Streamlit
