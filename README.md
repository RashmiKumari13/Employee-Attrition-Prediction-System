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

> For local development, frontend now loads `VITE_API_BASE_URL` from `frontend/.env.development` and will fall back to the relative `/api` path when the backend is served from the same host.
> 
> In production, you can also leave `frontend/.env.production` blank if the frontend and backend are deployed together behind the same origin. If the backend is deployed separately, set `VITE_API_BASE_URL` to that backend URL.
> 
> The backend now also supports serving the GitHub Pages production build asset path `/Employee-Attrition-Prediction-System/` when the frontend is built with that base path.

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

## Deployment Guide

### **Option 1: Deploy Frontend to GitHub Pages (Frontend Only)**

If you just want to deploy the frontend to GitHub Pages:

```bash
# From the repository root
cd frontend
npm install
npm run build
npm run deploy
```

This deploys only the frontend. **However, the app requires a backend API to function.**

### **Option 2: Full-Stack Deployment (Frontend + Backend)**

#### Step 1: Deploy Backend to Render.com (Recommended - Free)

1. Go to https://render.com/ and sign up
2. Create a new **Web Service**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `employee-attrition-api`
   - **Root Directory**: `backend`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `gunicorn app:app`
5. Set **Environment Variable**:
   - `PYTHON_VERSION` = `3.9`
6. Deploy

Wait for deployment to complete. Your backend URL will be something like: `https://employee-attrition-api.onrender.com`

#### Step 2: Update Frontend with Backend URL

1. Edit `frontend/.env.production`:
   ```
   VITE_API_BASE_URL=https://your-render-app.onrender.com
   ```

2. Commit and push:
   ```bash
   git add frontend/.env.production
   git commit -m "Update API endpoint for Render deployment"
   git push origin main
   ```

3. Deploy frontend to GitHub Pages:
   ```bash
   cd frontend
   npm install
   npm run build
   npm run deploy
   ```

#### Step 3: Verify Deployment

- Frontend: https://rashmikumari13.github.io/Employee-Attrition-Prediction-System/
- Backend: https://your-render-app.onrender.com/api/health

### **Option 3: Deploy Everything to Vercel (Experimental)**

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy to Vercel:
   ```bash
   vercel
   ```

3. Follow the prompts to connect your GitHub repository

**Note**: Vercel's Python support for serverless functions is limited. Render.com is recommended for this Flask API.

### Troubleshooting

**"Backend is up but model artifacts are missing"**
- Ensure `train_model.py` runs successfully during deployment
- Check that `backend/models/attrition_pipeline.joblib` is created
- Check that `backend/artifacts/model_metadata.json` is created

**"Cannot connect to API"**
- Verify backend service is running and healthy
- Check CORS is enabled on the backend (it is in `backend/app.py`)
- Verify `VITE_API_BASE_URL` in `frontend/.env.production` is correct

**CORS errors**
- Backend has `CORS` enabled for all origins - this is fine for demo purposes
- For production, restrict CORS to your frontend domain
