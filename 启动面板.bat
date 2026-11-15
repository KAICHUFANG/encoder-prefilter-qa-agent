@echo off
chcp 65001 >nul
cd /d "%~dp0"
set INTENT_DATASET=crosswoz_gate
set HF_HUB_OFFLINE=1
echo 面板地址: http://127.0.0.1:8010/dashboard
start "" http://127.0.0.1:8010/dashboard
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
