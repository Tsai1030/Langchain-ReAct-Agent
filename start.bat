@echo off
echo ========================================
echo 即時醫療資訊查詢系統 - 啟動腳本
echo ========================================
echo.

echo 正在檢查 Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo 錯誤：未找到 Node.js，請先安裝 Node.js
    pause
    exit /b 1
)

echo 正在安裝依賴...
call npm run install-all

echo.
echo 正在啟動開發伺服器...
echo 前端：http://localhost:3000
echo 後端：http://localhost:3001
echo.

call npm run dev

pause 