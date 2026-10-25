# Advanced Patient Readmission Risk Predictor

A machine learning project that predicts whether a patient will be readmitted to the hospital within 30 days of discharge, and presents the result as a risk score to support healthcare decisions.

## Project Pipeline

Patient Data → Data Preprocessing → Machine Learning Model → Readmission Risk Prediction → Healthcare Decision Support

## Tech Stack

| Component | Choice |
|---|---|
| Dataset | data in CSV |
| Data preparation | Pandas |
| Main ML model | Logistic Regression |
| Backend | FastAPI |
| Database | SQLite |
| Frontend | Streamlit |
| Collaboration | GitHub |

## Team Roles

| Person | Role | Responsibility |
|---|---|---|
| 1 | Data Person | Understand and clean the dataset, make graphs, prepare ML-ready data |
| 2 | ML Person | Train and compare models, evaluate and choose the final model, save the model |
| 3 | Backend Person | Build the API, connect the model, create the database and store predictions |
| 4 | Frontend Person | Build the patient form, result page and dashboard |
| 5 | Testing + Integration Person | Create the workflow, test everything, connect modules, final quality check |

## Repository Structure

```
hospital-readmission-predictor/
├── ml_dataset.csv          # Patient dataset
├── hospital_graphs.py      # Data exploration graphs
├── train_model.py          # ML training, evaluation and model saving
├── models/
│   ├── readmission_model.joblib   # Saved model bundle
│   └── final_metrics.json         # Final evaluation metrics
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

Backend and frontend files will be added by their owners.

## Getting Started

```bash
git clone https://github.com/<owner>/hospital-readmission-predictor.git
cd hospital-readmission-predictor
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

---

## Machine Learning Module

### Overview
This module trains a model that predicts whether a patient will be readmitted within 30 days of discharge. It outputs a risk probability, and the backend uses it to flag high-risk patients.

### Dataset
- File: `ml_dataset.csv` (1,000 patients, 13 columns, no missing values)
- Target: `readmitted_30_days` (1 = readmitted, 0 = not readmitted). About 42% of patients are readmitted.
- Features: age, gender, admission_type, diagnosis, time_in_hospital, previous_admissions, num_lab_procedures, num_medications, diabetes, medication_change, discharge_disposition
- `patient_id` is dropped before training because it is not a feature.

### Approach
1. Split the data 80% train / 20% test, stratified on the target.
2. Preprocess inside a scikit-learn Pipeline: one-hot encode the categorical columns and scale the numeric columns for Logistic Regression.
3. Train two models and compare them:
   - **Logistic Regression** (final model)
   - **Random Forest** (comparison)
4. Evaluate with accuracy, precision, recall, F1, ROC-AUC, the confusion matrix and 5-fold cross-validation.
5. Tune the decision threshold on the training data only, so that recall stays at 70% or more while keeping the best F1.
6. Refit the final model on all the data and save it.

### Results

| Model | Test accuracy | Test recall (0.5) | Test AUC |
|---|---|---|---|---|
| Logistic Regression | 0.605 | 0.560 | 0.646 | **0.714** |

**Final model: Logistic Regression with threshold 0.36**
- Accuracy 0.58, precision 0.50, recall 0.82, F1 0.62 on the test set
- It catches 69 of 84 readmitted patients and misses 15. The cost is 69 false alarms.

### Why Logistic Regression
- It has the best cross-validated ROC-AUC.
- It outputs calibrated risk probabilities, which suit a risk-prediction app.
- Its coefficients are easy to explain to clinicians. The strongest risk factors are age, Rehabilitation discharge, diabetes, medication change and Emergency admission.
- With only 1,000 rows and mostly linear, weak patterns, a simple model generalizes better than complex ones.
- The threshold of 0.36 favors recall, because missing a high-risk patient is worse than a false alarm.

### Limitations
- The dataset and the signal is weak, so accuracy near 67% and AUC near 0.71 is the realistic ceiling.
- The test split is only 200 rows, so single-split numbers are noisy. Use the cross-validation numbers as the main reference.
- The model is for decision support only and does not replace clinical judgment.

### How to train
```bash
python train_model.py
```
This creates `models/readmission_model.joblib` and `models/final_metrics.json`.

### How to use the saved model
```python
import joblib, pandas as pd

bundle = joblib.load("models/readmission_model.joblib")
patient = pd.DataFrame([{
    "age": 70, "gender": "Male", "admission_type": "Emergency",
    "diagnosis": "Heart Disease", "time_in_hospital": 8,
    "previous_admissions": 3, "num_lab_procedures": 50,
    "num_medications": 20, "diabetes": "Yes",
    "medication_change": "Yes", "discharge_disposition": "Rehabilitation",
}])

risk = bundle["pipeline"].predict_proba(patient[bundle["feature_columns"]])[:, 1][0]
high_risk = risk >= bundle["threshold"]
print(f"Readmission risk: {risk:.0%} | High risk: {high_risk}")
```
Use the same scikit-learn version as in `requirements.txt`, otherwise loading the model may fail.
