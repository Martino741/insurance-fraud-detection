# 🚗 Automobile Insurance Fraud Detection

A machine learning project that identifies automobile insurance claims that may require fraud investigation.

The system is designed to support human investigators by prioritizing suspicious claims. It does not make automatic fraud decisions.

## Demo

![Streamlit application demo]:
Insurance fraud can generate substantial financial losses. The objective of this project is to classify automobile insurance claims into two categories:

- `0` — low fraud risk
- `1` — claim requiring further investigation

## Dataset

The project uses a public automobile insurance claims dataset containing 1,000 records and 39 relevant variables.

The target column is:

```text
fraud_reported


- N is encoded as 0
- Y is encoded as 1

The dataset includes policy details, customer information, accident information, claim amounts, witnesses, police reports, vehicle information, and the fraud label.
Dataset source: Insurance Claims Dataset
Data Preparation
The following preprocessing steps were performed:
- Removed the empty _c39 column.
- Handled missing values in authorities_contacted.
- Replaced unknown values represented by ?.
- Encoded categorical variables using one-hot encoding.
- Converted fraud_reported into a binary target.
- Extracted useful date components.
- Split data into training and test sets using stratified sampling.
- Used RobustScaler to reduce the influence of outliers.
- Applied class_weight="balanced" to address the class imbalance.
Models Evaluated
Three classification models were evaluated:
Model	Accuracy	Precision (Fraud)	Recall (Fraud)	F1-score (Fraud)
Logistic Regression	78.0%	53.7%	73.5%	62.1%
Support Vector Classifier	82.5%	62.1%	73.5%	67.3%
Random Forest	78.5%	62.5%	30.6%	41.1%


Final Model
The Support Vector Classifier (SVC) was selected as the final model.
It achieved:
- Accuracy: 82.5%
- Precision for fraud: 62.1%
- Recall for fraud: 73.5%
- F1-score for fraud: 67.3%
On the test set, the model detected 36 out of 49 fraudulent claims.


Application
A Streamlit application allows users to:
- analyse an individual insurance claim;
- analyse a prepared CSV file containing multiple claims;
- identify claims requiring additional investigation;
- export prediction results as a CSV file.

Project Structure

insurance-fraud-detection/
├── data/
│   ├── insurance_claims.csv
│   └── processed_claim_features.csv
├── images/
│   └── app-demo.png
├── models/
│   └── insurance_fraud_model.joblib
├── notebooks/
│   └── 01_insurance_fraud_analysis.ipynb
├── app.py
├── requirements.txt
└── README.md


Limitations
- The dataset is relatively small, with 1,000 claims.
- The model should be used as a decision-support tool only.
- Predictions must be reviewed by qualified insurance investigators.
- The current application expects a CSV with the same prepared features used during model training.

Author
Martino VIGNIGBE

[def]: https://github.com/Martino741/insurance-fraud-detection/blob/5bcc51e4e97364f9c5f598eda61581b243f4fa3f/images/Image.png1.jpeg