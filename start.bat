@echo off
echo === Starting FastAPI backend ===
start cmd /k "uvicorn fast-streamlit:app --host 0.0.0.0 --port 8000"

timeout /t 3 > nul

echo === Starting Streamlit frontend ===
start cmd /k "npm run dev"

echo === Starting Streamlit Training app === 
start cmd /k "cd streamlit-app && streamlit run Re-training_app.py"

echo All servers launched! Check the URLs in your browser.
pause