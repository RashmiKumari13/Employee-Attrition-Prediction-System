# Employee Attrition Prediction System

Full-stack AI application that predicts employee attrition risk using the **IBM HR Analytics dataset**, a **Gradient Boosting classifier**, a **Flask REST API**, and a **React.js dashboard** with **SHAP explainability**.

## Tech Stack

- Python
- Scikit-learn
- Flask + Flask-CORS
- SHAP
- ReportLab (server PDF reports)
- React.js (Vite)
- jsPDF (client PDF reports)
- REST API

## Core Features

- End-to-end binary attrition prediction (`Yes`/`No`) with probability score
- Gradient Boosting training pipeline targeting ~**89% accuracy**
- Model metadata and metrics endpoint
- Dynamic frontend form generated from backend feature schema
- SHAP-based explanation report for each prediction (top positive/negative contributors)
- Downloadable PDF explainability reports with risk score, top factors, SHAP graph, and retention recommendations
- Clean backend/frontend separation for production-ready extension

## Project Structure

```text
Employee-Attrition-Prediction-main/
|-- backend/
|   |-- app.py
|   |-- config.py
|   |-- model_service.py
|   |-- report_service.py
|   |-- train_model.py
|   |-- requirements.txt
|   |-- data/
|   |   `-- WA_Fn-UseC_-HR-Employee-Attrition.csv
|   |-- models/
|   |   `-- attrition_pipeline.joblib         # generated after training
|   `-- artifacts/
|       `-- model_metadata.json               # generated after training
|-- frontend/
|   |-- package.json
|   |-- vite.config.js
|   |-- index.html
|   `-- src/
|       |-- App.jsx
|       |-- main.jsx
|       `-- styles.css
`-- README.md
```

## Setup and Run

### 1) Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

### 2) Train model (creates artifacts)

```bash
python train_model.py
```

This generates:

- `backend/models/attrition_pipeline.joblib`
- `backend/artifacts/model_metadata.json`

### 3) Start Flask API

```bash
python app.py
```

Default API URL: `http://127.0.0.1:5000`

### 4) Frontend setup

```bash
cd ../frontend
npm install
npm run dev
```

Default frontend URL: `http://127.0.0.1:5173`

## API Endpoints

- `GET /api/health` -> service health + model readiness
- `GET /api/model-info` -> model + metrics metadata
- `GET /api/features` -> input schema used by the React form
- `POST /api/predict` -> attrition prediction + probability + SHAP report
- `POST /api/report` -> downloadable ReportLab PDF explainability report
- `POST /api/reload-model` -> reload model artifacts without restarting server

`POST /api/report` accepts the same JSON payload as `POST /api/predict` and returns a PDF attachment.

## Example Prediction Request

```json
{
  "Age": 35,
  "BusinessTravel": "Travel_Rarely",
  "DailyRate": 1024,
  "Department": "Research & Development",
  "DistanceFromHome": 7,
  "Education": 3,
  "EducationField": "Life Sciences",
  "EnvironmentSatisfaction": 3,
  "Gender": "Female",
  "HourlyRate": 65,
  "JobInvolvement": 3,
  "JobLevel": 2,
  "JobRole": "Research Scientist",
  "JobSatisfaction": 3,
  "MaritalStatus": "Married",
  "MonthlyIncome": 6200,
  "MonthlyRate": 14000,
  "NumCompaniesWorked": 2,
  "OverTime": "No",
  "PercentSalaryHike": 13,
  "PerformanceRating": 3,
  "RelationshipSatisfaction": 3,
  "StockOptionLevel": 1,
  "TotalWorkingYears": 10,
  "TrainingTimesLastYear": 2,
  "WorkLifeBalance": 3,
  "YearsAtCompany": 5,
  "YearsInCurrentRole": 3,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 4
}
```

## Example Response

```json
{
  "prediction": 0,
  "risk_label": "Low Risk",
  "attrition_probability": 0.1824,
  "attrition_percent": 18.24,
  "warnings": [],
  "summary": "Employee likely to leave due to Over Time + Job Satisfaction.",
  "recommendations": [
    "Reduce recurring overtime through workload balancing and manager check-ins.",
    "Create a role-specific engagement plan to improve job satisfaction."
  ],
  "input_features": {
    "...": "..."
  },
  "explainability": {
    "base_value": -1.34917,
    "raw_output_estimate": -1.47821,
    "top_contributors": [
      {
        "feature": "OverTime",
        "shap_value": 0.23342,
        "direction": "increase_risk"
      },
      {
        "feature": "MonthlyIncome",
        "shap_value": -0.16421,
        "direction": "decrease_risk"
      }
    ]
  }
}
```

## Notes

- The Flask API returns `503` on model endpoints until training artifacts exist.
- Unknown categorical values are accepted and flagged in response warnings.
- SHAP values are generated from the trained tree model for per-user explainability.
- Frontend provides two export options: browser-side PDF (`jsPDF`) and server-side PDF (`ReportLab`).
