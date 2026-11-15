@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo  客服意图识别 Agent 启动中……
echo  接口文档： http://127.0.0.1:8010/docs
echo  健康检查： http://127.0.0.1:8010/health
echo  关闭本窗口即停止服务
echo ============================================
uv run uvicorn app.main:app --reload --port 8010
pause