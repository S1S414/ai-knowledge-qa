@echo off
chcp 65001 >nul
title 知识库问答系统 - Port 5002
cd /d "%~dp0"
echo ================================
echo   知识库问答系统 - RAG架构
echo   访问地址: http://localhost:5002
echo ================================
echo.
streamlit run app.py --server.port 5002
pause
