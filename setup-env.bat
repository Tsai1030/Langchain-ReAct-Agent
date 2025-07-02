@echo off
echo ========================================
echo 醫療資訊查詢系統 - 環境變數設定
echo ========================================
echo.

echo 正在建立環境變數檔案...
echo.

REM 檢查 .env 檔案是否存在
if exist "server\.env" (
    echo 發現現有的 .env 檔案
    echo 是否要覆蓋？(Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo 覆蓋現有檔案...
    ) else (
        echo 取消操作
        pause
        exit /b 0
    )
)

REM 建立 .env 檔案
echo # OpenAI API 設定 > server\.env
echo OPENAI_API_KEY=your_openai_api_key_here >> server\.env
echo. >> server\.env
echo # ScrapingBee API 設定 (動態網頁抓取) >> server\.env
echo SCRAPING_BEE_KEY=your_scraping_bee_api_key_here >> server\.env
echo. >> server\.env
echo # Serper API 設定 (Google 搜尋) >> server\.env
echo SERPER_API_KEY=your_serper_api_key_here >> server\.env
echo. >> server\.env
echo # 伺服器設定 >> server\.env
echo PORT=3001 >> server\.env
echo NODE_ENV=development >> server\.env
echo. >> server\.env
echo # 可選設定 >> server\.env
echo CORS_ORIGIN=http://localhost:3000 >> server\.env

echo ✅ 環境變數檔案已建立：server\.env
echo.
echo 📝 請編輯 server\.env 檔案，填入您的 API 金鑰：
echo.
echo 1. 前往 https://platform.openai.com/ 取得 OpenAI API Key
echo 2. 前往 https://www.scrapingbee.com/ 取得 ScrapingBee API Key
echo 3. 前往 https://serper.dev/ 取得 Serper API Key (可選)
echo 4. 將金鑰填入對應的變數中
echo.
echo 範例：
echo OPENAI_API_KEY=sk-1234567890abcdef...
echo SCRAPING_BEE_KEY=your-scraping-bee-key-here
echo SERPER_API_KEY=your-serper-key-here
echo.
echo 設定完成後，請重新執行 start.bat
echo.
pause 