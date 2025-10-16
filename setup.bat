@echo off
echo === Running setup ===

REM Run PowerShell installer (Python, Node.js, PostgreSQL)
powershell -ExecutionPolicy Bypass -File install.ps1

echo === Waiting for environment setup to complete... ===

set "PATH=%PATH%;%ProgramFiles%\nodejs"

where node >nul 2>nul
IF ERRORLEVEL 1 (
    echo Node.js is not installed or not in PATH. Please restart your terminal or install manually.
    pause
    exit /b 1
)

IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing Python dependencies...
python -m pip install -r requirements.txt

echo Installing Node dependencies...
npm install --force --legacy-peer-deps >> npm_log.txt 2>&1
IF ERRORLEVEL 1 (
    echo npm install failed! Check npm_log.txt
    pause
    exit /b 1
)

echo === Setup completed! You can now run start.bat ===
pause
