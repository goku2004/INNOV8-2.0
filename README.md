
# Predicting Troop Betrayal: Machine Learning Pipeline

## Introduction

This repository contains the implementation of a machine learning pipeline aimed at predicting troop betrayal in the war against the Phrygians. The goal of the project is to build a system that analyzes various factors to predict the likelihood of betrayal among soldiers. The project utilizes multiple machine learning models and an ensemble approach to improve prediction accuracy.

## Table of Contents
1. Introduction
2. Dataset
3. Pipeline Overview
4. Dependencies
5. Setup Instructions
6. Results and Evaluation
7. Future Improvements

---

## 1. Dataset

The dataset, `extended_dummy_troop_betrayal_dataset.csv`, contains features related to soldier profiles, such as salary satisfaction, bonus expectations, peer recognition, and more. The target variable is `Betrayal`, where `1` represents a soldier who is likely to betray, and `0` represents loyalty to the clan.

### Dataset Columns:
- `Salary_Satisfaction_Score`
- `Bonus_Expectations`
- `Phrygian_Offer_Exposure`
- `Attraction_to_Phrygian_Ideals`
- ... (more features related to soldier behavior and financial status)
- `Betrayal` (Target)

---

## 2. Pipeline Overview

The machine learning pipeline follows these steps:

### **Step 1: Data Ingestion**
- The dataset is loaded into a Pandas DataFrame from a CSV file.
  
### **Step 2: Data Preprocessing**
- Features are selected, and the target variable (`Betrayal`) is separated.
- Data is cleaned and checked for missing values.
  
### **Step 3: Feature Selection**
- All columns except the target column (`Betrayal`) are selected as features (`X`), and `y` represents the target.

### **Step 4: Train-Test Split**
- The data is split into training (80%) and testing (20%) sets using `train_test_split`.

### **Step 5: Handling Imbalanced Classes**
- SMOTE (Synthetic Minority Oversampling Technique) is used to handle class imbalance, oversampling the minority class (betrayal).

### **Step 6: Model Training**
- Several classifiers are trained, including:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost
  - LightGBM
  - Logistic Regression
  - Support Vector Classifier (SVC)

### **Step 7: Ensemble Method**
- A Voting Classifier is used to combine the predictions of multiple models to improve overall accuracy.

### **Step 8: Model Evaluation**
- The performance of the models is evaluated using `accuracy_score` on the test set.

### **Step 9: Hyperparameter Tuning**
- Grid Search with cross-validation (`GridSearchCV`) is used to tune model hyperparameters for optimal performance.

### **Step 10: Prediction**
- Final predictions are made on the test set using the best-performing model.

## 3. Dependencies

Ensure that the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `imblearn` (for SMOTE)


## 5. Results and Evaluation

The pipeline outputs accuracy scores for each model used. The best model is selected based on the highest accuracy achieved.

---

## 6. Future Improvements

- Incorporate additional metrics like precision, recall, and F1-score to provide a more comprehensive evaluation of the model performance.
- Explore advanced ensemble methods like Stacking.
- Add more robust feature engineering techniques such as interaction terms or domain-specific features.
- Expand the dataset and explore real-world factors to improve the prediction of betrayal.

---

