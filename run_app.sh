#!/usr/bin/env bash
# ============================================================================
# AnatomoZK: 3D Coronary Risk & Byzantine BFT Trust
# One-Click Reproducible Production Launcher (Linux / macOS)
# ============================================================================

set -e

echo "============================================================================"
echo "  ANATOMOZK // 3D CORONARY RISK & BYZANTINE BFT TRUST"
echo "  Automated Pre-Flight Verification & Clinical WebGL Dashboard Launcher"
echo "============================================================================"
echo ""

# 1. Verify Python Installation
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# 2. Run Pre-Flight Verification Suite
echo "[Step 1/2] Executing Pre-Flight Suitability Ledger..."
$PYTHON_CMD check_assets.py || echo "[WARNING] Pre-flight check returned non-zero status. Proceeding..."

echo ""
echo "[Step 2/2] Launching Clinical WebGL Application on http://localhost:8501 ..."
$PYTHON_CMD -m streamlit run app.py --server.headless false --server.port 8501
