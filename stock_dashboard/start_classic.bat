@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 安裝/更新套件中...
python -m pip install -q --no-warn-script-location -r requirements.txt
echo 經典桌面版：http://localhost:8501
python -m streamlit run app.py
pause
