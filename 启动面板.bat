@echo off
chcp 65001 >nul
cd /d "%~dp0"
set INTENT_DATASET=crosswoz_gate_full
set INTENT_MAX_LEN=64
set HF_HUB_OFFLINE=1
echo 正在启动服务（首次加载 4 个模型约需 10 秒）...
echo 面板地址: http://127.0.0.1:8010/dashboard
start "" /min .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
timeout /t 6 >nul
start "" http://127.0.0.1:8010/dashboard
