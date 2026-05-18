@echo off
REM Deploy Frontend to GitHub Pages
cd /d "f:\GITHUB\Employee-Attrition-Prediction-System\frontend"

echo Installing dependencies...
call npm install

echo Building frontend...
call npm run build

echo Deploying to GitHub Pages...
call npm run deploy

echo.
echo ✅ Deployment complete! Your site is live at:
echo https://rashmikumari13.github.io/Employee-Attrition-Prediction-System/
pause
