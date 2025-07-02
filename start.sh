#!/bin/bash

echo "========================================"
echo "即時醫療資訊查詢系統 - 啟動腳本"
echo "========================================"
echo

# 檢查 Node.js
if ! command -v node &> /dev/null; then
    echo "錯誤：未找到 Node.js，請先安裝 Node.js"
    exit 1
fi

echo "正在安裝依賴..."
npm run install-all

echo
echo "正在啟動開發伺服器..."
echo "前端：http://localhost:3000"
echo "後端：http://localhost:3001"
echo

npm run dev 