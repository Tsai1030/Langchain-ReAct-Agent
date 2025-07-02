#!/bin/bash

echo "========================================"
echo "醫療資訊查詢系統 - 環境變數設定"
echo "========================================"
echo

echo "正在建立環境變數檔案..."
echo

# 檢查 .env 檔案是否存在
if [ -f "server/.env" ]; then
    echo "發現現有的 .env 檔案"
    read -p "是否要覆蓋？(y/N): " choice
    if [[ $choice =~ ^[Yy]$ ]]; then
        echo "覆蓋現有檔案..."
    else
        echo "取消操作"
        exit 0
    fi
fi

# 建立 .env 檔案
cat > server/.env << EOF
# OpenAI API 設定
OPENAI_API_KEY=your_openai_api_key_here

# Serper API 設定 (Google 搜尋)
SERPER_API_KEY=your_serper_api_key_here

# 伺服器設定
PORT=3001
NODE_ENV=development

# 可選設定
CORS_ORIGIN=http://localhost:3000
EOF

echo "✅ 環境變數檔案已建立：server/.env"
echo
echo "📝 請編輯 server/.env 檔案，填入您的 API 金鑰："
echo
echo "1. 前往 https://platform.openai.com/ 取得 OpenAI API Key"
echo "2. 前往 https://serper.dev/ 取得 Serper API Key"
echo "3. 將金鑰填入對應的變數中"
echo
echo "範例："
echo "OPENAI_API_KEY=sk-1234567890abcdef..."
echo "SERPER_API_KEY=your-serper-key-here"
echo
echo "設定完成後，請重新執行 start.sh"
echo 