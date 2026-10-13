@echo off
REM ============================================================================
REM AnatomoZK: 3D Coronary Risk & Byzantine BFT Trust
REM One-Click Reproducible Production Launcher (Windows)
REM ============================================================================

echo ============================================================================
echo   ANATOMOZK // 3D CORONARY RISK ^& BYZANTINE BFT TRUST
echo   Automated Pre-Flight Verification ^& Clinical WebGL Dashboard Launcher
echo ============================================================================
echo.

REM 1. Verify Python Installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not detected in system PATH.
    echo Please install Python 3.10+ and re-run.
    pause
    exit /b 1
)

REM 2. Run Pre-Flight Verification Suite
echo [Step 1/2] Executing Pre-Flight Suitability Ledger...
python check_assets.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Pre-flight asset check flagged warnings.
    echo Proceeding with execution...
    echo.
) else (
    echo.
    echo [Step 1/2 Complete] Pre-Flight Clearance 100%% Verified.
    echo.
)

REM 3. Launch Streamlit Clinical WebGL Dashboard
echo [Step 2/2] Launching Clinical WebGL Application on http://localhost:8501 ...
python -m streamlit run app.py --server.headless false --server.port 8501

pause
