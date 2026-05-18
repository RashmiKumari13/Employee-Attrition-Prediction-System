# Quick Deploy to GitHub Pages

## Option 1: Run the Deploy Script (Easiest)
Double-click: `deploy-frontend.bat`

This will:
1. Install dependencies
2. Build the React app
3. Deploy to GitHub Pages

---

## Option 2: Manual Deployment (Windows Command Prompt)

```batch
cd frontend
npm install
npm run build
npm run deploy
```

---

## Option 3: Step-by-Step in PowerShell

```powershell
cd frontend
npm install
npm run build
npm run deploy
```

---

## After Deployment

✅ Your site will be live at:
**https://rashmikumari13.github.io/Employee-Attrition-Prediction-System/**

⚠️ **Important**: The app needs a backend API to work with predictions.

To make predictions work:

1. Deploy backend to Render.com (free):
   - https://render.com → New Web Service
   - Root Directory: `backend`
   - Build: `pip install -r requirements.txt && python train_model.py`
   - Start: `gunicorn app:app`
   - Copy your URL (e.g., https://employee-attrition-api.onrender.com)

2. Update `frontend/.env.production`:
   ```
   VITE_API_BASE_URL=https://your-render-url.onrender.com
   ```

3. Redeploy:
   ```
   npm run deploy
   ```

See DEPLOYMENT.md for complete instructions.
