# 即時醫療資訊查詢系統 - 設定指南

## 🚀 快速開始

### 1. 系統需求
- Node.js 18+ 
- npm 或 yarn
- OpenAI API Key
- Serper API Key

### 2. 取得 API 金鑰

#### OpenAI API Key
1. 前往 [OpenAI Platform](https://platform.openai.com/)
2. 註冊/登入帳號
3. 前往 API Keys 頁面
4. 建立新的 API Key
5. 複製並保存金鑰

#### Serper API Key (Google 搜尋)
1. 前往 [Serper.dev](https://serper.dev/)
2. 註冊帳號
3. 選擇適合的方案 (免費版每月 100 次搜尋)
4. 取得 API Key

### 3. 安裝與設定

#### 方法一：使用啟動腳本 (推薦)
```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh
./start.sh
```

#### 方法二：手動安裝
```bash
# 1. 安裝所有依賴
npm run install-all

# 2. 設定環境變數
cp server/env.example server/.env
# 編輯 server/.env 檔案，填入您的 API 金鑰

# 3. 啟動開發伺服器
npm run dev
```

### 4. 環境變數設定

在 `server/.env` 檔案中設定：

```env
# OpenAI API 設定
OPENAI_API_KEY=sk-your-openai-api-key-here

# Serper API 設定 (Google 搜尋)
SERPER_API_KEY=your-serper-api-key-here

# 伺服器設定
PORT=3001
NODE_ENV=development
```

### 5. 驗證安裝

1. 開啟瀏覽器前往 http://localhost:3000
2. 在查詢框中輸入：「今天高醫心臟內科看到幾號？」
3. 點擊搜尋按鈕
4. 等待系統回應

## 🔧 進階設定

### 自訂搜尋範圍
在 `server/services/queryService.js` 中修改：

```javascript
// 修改搜尋參數
const response = await axios.post(this.baseURL, {
  q: query,
  num: 10,        // 搜尋結果數量
  gl: 'tw',       // 地區設定
  hl: 'zh-tw'     // 語言設定
}, {
  // ...
});
```

### 調整 AI 回應
在 `server/services/queryService.js` 中修改 GPT 提示詞：

```javascript
const prompt = `你是一個專業的醫療資訊查詢助手...
// 自訂您的提示詞
`;
```

### 修改前端樣式
在 `client/src/index.css` 中自訂 Tailwind CSS 樣式：

```css
@layer components {
  .btn-primary {
    @apply bg-medical-600 hover:bg-medical-700;
  }
}
```

## 🐛 故障排除

### 常見問題

#### 1. API 金鑰錯誤
```
錯誤：API 設定錯誤
解決方案：檢查 .env 檔案中的 API 金鑰是否正確
```

#### 2. 網路連線問題
```
錯誤：搜尋服務錯誤
解決方案：檢查網路連線和 Serper API 狀態
```

#### 3. 前端無法連線後端
```
錯誤：網路連線錯誤
解決方案：確認後端伺服器在 http://localhost:3001 運行
```

#### 4. 依賴安裝失敗
```bash
# 清除快取重新安裝
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### 日誌查看
- 前端日誌：瀏覽器開發者工具 Console
- 後端日誌：終端機輸出

## 📊 監控與維護

### 健康檢查
```bash
curl http://localhost:3001/health
```

### API 使用量監控
- OpenAI：在 OpenAI Platform 查看使用量
- Serper：在 Serper Dashboard 查看搜尋次數

### 效能優化
1. 調整搜尋結果數量 (`num` 參數)
2. 優化 GPT 提示詞長度
3. 實作快取機制 (可選)

## 🔒 安全性建議

1. **API 金鑰保護**
   - 永遠不要將 `.env` 檔案提交到 Git
   - 使用環境變數管理敏感資訊

2. **速率限制**
   - 系統已內建速率限制 (15分鐘100次請求)
   - 可根據需求調整

3. **輸入驗證**
   - 查詢長度限制：500字元
   - 內容過濾：可實作額外的內容檢查

## 🚀 部署指南

### 生產環境部署
1. 設定 `NODE_ENV=production`
2. 使用 PM2 或 Docker 部署
3. 設定反向代理 (Nginx)
4. 啟用 HTTPS

### Docker 部署 (可選)
```dockerfile
# 建立 Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3001
CMD ["npm", "start"]
```

## 📞 支援

如有問題，請檢查：
1. API 金鑰是否正確
2. 網路連線是否正常
3. 伺服器日誌是否有錯誤訊息
4. 瀏覽器 Console 是否有錯誤

---

**注意**：本系統僅供參考，不構成醫療建議。請以醫院官方資訊為準。 