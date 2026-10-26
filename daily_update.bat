@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 給 Windows「工作排程器」每天收盤後執行，不開畫面也能累積資料與重新訓練
python update.py >> update_log.txt 2>&1
