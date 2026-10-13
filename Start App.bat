@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\streamlit.exe" (
    echo First-time setup: installing dependencies, please wait...
    python -m venv venv
    venv\Scripts\python -m pip install --upgrade pip
    venv\Scripts\python -m pip install -r requirements.txt
)
if not exist "certs\cert.pem" (
    echo Creating security certificate for the camera...
    venv\Scripts\python make_cert.py
)
echo Trusting local certificate for this PC (one-time)...
venv\Scripts\python -c "import subprocess; subprocess.run(['certutil','-user','-addstore','Root','certs\\cert.pem'], capture_output=True)"
start "" cmd /c "timeout /t 10 >nul & start "" https://localhost:8501"
echo.
echo  Road Anomaly Detection is starting...
echo  On this PC : https://localhost:8501
echo  On phone   : open Home page in the app to see the phone link (same Wi-Fi)
echo  (close this window to stop the app)
echo.
venv\Scripts\python -m streamlit run main.py --server.headless true --server.port 8501 --server.sslCertFile certs\cert.pem --server.sslKeyFile certs\key.pem
