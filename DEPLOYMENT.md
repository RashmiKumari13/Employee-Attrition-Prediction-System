# Deployment Guide: Employee Attrition Prediction System

## Overview

This is a full-stack application:
- **Frontend**: React app deployed to GitHub Pages (static hosting)
- **Backend**: Flask API that must be deployed separately to a service that supports Python

GitHub Pages can only host static files - it **cannot run a Python Flask server**. Therefore, the backend must be deployed to a separate service.

## Recommended Deployment: Render.com (Free Tier)

### Step-by-Step Instructions

#### 1. Deploy Backend to Render.com

1. Go to https://render.com and create a free account
2. Click **New +** → **Web Service**
3. Select **GitHub** and authorize
4. Find and select your `Employee-Attrition-Prediction-System` repository
5. Fill in the configuration:

   | Field | Value |
   |-------|-------|
   | **Name** | `employee-attrition-api` |
   | **Root Directory** | `backend` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt && python train_model.py` |
   | **Start Command** | `gunicorn app:app` |

6. **Add Environment Variables**:
   - Click **Add Environment Variable**
   - Key: `PYTHON_VERSION`
   - Value: `3.9`

7. Click **Create Web Service**

⏳ **Wait 5-10 minutes** for the build to complete. Once done, you'll see a URL like:
```
https://employee-attrition-api.onrender.com
```

**Test the backend**: Visit https://employee-attrition-api.onrender.com/api/health

#### 2. Update Frontend Configuration

1. Edit `frontend/.env.production` and replace with your Render URL:
   ```
   VITE_API_BASE_URL=https://employee-attrition-api.onrender.com
   ```

2. Save and commit:
   ```bash
   git add frontend/.env.production
   git commit -m "Update API endpoint to Render backend"
   git push origin main
   ```

#### 3. Deploy Frontend to GitHub Pages

```bash
cd frontend
npm install
npm run build
npm run deploy
```

✅ Your app is now live at:
- **Frontend**: https://rashmikumari13.github.io/Employee-Attrition-Prediction-System/
- **Backend**: https://employee-attrition-api.onrender.com

---

## Alternative Deployments

### Option A: Railway.app (Free tier - recommended alternative)

Similar process to Render.com:

1. Go to https://railway.app
2. Create a new project from your GitHub repository
3. Select the `backend` directory as root
4. Add build command: `pip install -r requirements.txt && python train_model.py`
5. Add start command: `gunicorn app:app`
6. Railway will assign you a domain

Update `frontend/.env.production` with the Railway URL and redeploy frontend.

### Option B: Heroku (Paid - $50/month minimum)

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create backend app
heroku create employee-attrition-api --buildpack heroku/python

# Deploy backend
cd backend
git subtree push --prefix backend heroku main

# Update frontend with Heroku URL
# Then deploy frontend to GitHub Pages
```

---

## Troubleshooting

### Frontend loads but shows error message

**Error**: "Backend is up but model artifacts are missing"
- ✅ **Solution**: Backend deployed but training didn't complete
  - Check backend logs on Render/Railway
  - Training takes 2-5 minutes. Wait longer or restart the service
  - SSH into backend and run: `python train_model.py`

**Error**: "Cannot connect to backend API"
- ✅ **Solution**: Wrong URL or backend is down
  - Verify URL in `frontend/.env.production` is correct
  - Test endpoint manually: `curl https://your-backend.com/api/health`
  - Restart the backend service

**Error**: "Failed to fetch"
- ✅ **Solution**: CORS or network issue
  - Verify backend has CORS enabled (it does by default)
  - Check frontend `.env.production` has correct API URL
  - Rebuild and redeploy frontend: `npm run deploy`

### Backend takes too long to start

- Render.com free tier uses shared resources - first startup can take 2-5 minutes
- Once running, subsequent requests are faster
- If "Failed to fetch" appears after 10 minutes, restart the service on Render

### Cold start issues

Render.com spins down free tier apps after 15 minutes of inactivity. First request after spin-down takes 30+ seconds.

**Solution**: 
- Use a paid tier for production
- Or ping the health endpoint every 10 minutes to keep it warm

---

## Local Development

To test both frontend and backend locally before deploying:

```bash
# Terminal 1: Start Backend
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows: or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python train_model.py
python app.py

# Terminal 2: Start Frontend
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 - frontend proxies API calls to http://localhost:5000

---

## Environment Variables Reference

### Frontend (`frontend/.env.production`)
```
VITE_API_BASE_URL=https://your-backend-url.com
```

### Backend (Render.com Environment Variables)
```
PYTHON_VERSION=3.9
```

---

## Production Best Practices

1. **Pin Python version** in backend requirements
2. **Use gunicorn** in production (not Flask dev server)
3. **Enable CORS properly** - restrict to your frontend domain:
   ```python
   CORS(app, resources={
       r"/api/*": {"origins": "https://your-frontend-domain.github.io"}
   })
   ```
4. **Add rate limiting** to prevent abuse
5. **Monitor error logs** on your deployment platform
6. **Set up automatic redeploys** when you push to GitHub

