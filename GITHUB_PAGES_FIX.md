# GitHub Pages Deployment Fix - Quick Start

## The Problem ⚠️

Your site deployed to GitHub Pages appears broken because:
- GitHub Pages **only hosts static files** (HTML, CSS, JS)
- Your React app requires a **Flask backend API** to work
- The backend doesn't exist on GitHub Pages, so the app can't load data

**Locally it works because**: Your dev server proxies `/api/*` calls to `http://127.0.0.1:5000`

## The Solution ✅

Deploy your Flask backend to a service that supports Python, then tell your frontend where to find it.

### Quick Fix (2 steps):

#### 1️⃣ Deploy Backend to Render.com (3 minutes)

1. Go to https://render.com (free account)
2. Click: **New** → **Web Service**
3. Connect GitHub, select your repo
4. Set these values:
   - **Name**: `employee-attrition-api`
   - **Root Directory**: `backend`
   - **Build**: `pip install -r requirements.txt && python train_model.py`
   - **Start**: `gunicorn app:app`

5. Click **Create**
6. ⏳ Wait 5-10 minutes, copy your URL (e.g., `https://employee-attrition-api.onrender.com`)

#### 2️⃣ Update Frontend & Redeploy (2 minutes)

```bash
# Edit this file and replace the URL with your Render URL from step 1
frontend/.env.production
# Change to: VITE_API_BASE_URL=https://your-url.onrender.com

# Commit and deploy
cd frontend
npm install
npm run build
npm run deploy
```

✅ **Done!** Your site will now work: https://rashmikumari13.github.io/Employee-Attrition-Prediction-System/

---

## Full Instructions

See **DEPLOYMENT.md** for detailed step-by-step guide with troubleshooting.

## Alternative Platforms

- **Railway.app** (similar process, also free)
- **Heroku** (paid option)
- See DEPLOYMENT.md for complete instructions
