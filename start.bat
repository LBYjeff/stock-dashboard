@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] 安裝/更新套件中...
python -m pip install -q --no-warn-script-location -r requirements.txt
echo.
echo [2/3] 每日更新：抓新資料、重新訓練模型（第一次建立資料庫每檔約 10～30 秒）
python update.py
echo.
echo [3/3] 啟動美台股看盤（iOS 介面版）
echo   電腦：http://localhost:8501
echo   手機/平板：連同一個 Wi-Fi，開下方 Network URL，再「加入主畫面」
echo   第一次若跳出 Windows 防火牆，請按「允許存取」。關閉此視窗即停止。
echo.
python -m streamlit run stock_app.py --server.address 0.0.0.0 --server.port 8501
pause
