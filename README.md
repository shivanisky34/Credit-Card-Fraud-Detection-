# Credit-Card-Fraud-Detection-
Machine Learning model for detecting credit card fraud using rare-event classification.
# Credit Card Fraud Detection Engine

A Machine Learning project focused on detecting fraudulent credit card transactions. 
Fraud detection is a **rare-event classification** problem with extreme class imbalance (<0.2% fraudulent transactions).

## 🛠️ Software Stack
- **Language:** Python 3
- **Machine Learning:** Scikit-Learn (RandomForestClassifier)
- **Data Engineering:** Pandas, NumPy

## 📊 Feature Architecture
- **V1 - V28:** PCA transformed confidential transaction features
- **Amount_Scaled / Time_Scaled:** Standardized transaction metadata
- **Target Variable:** `Class` (0: Normal Transaction, 1: Fraudulent Transaction)
