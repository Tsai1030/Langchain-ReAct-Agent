const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const queryRoutes = require('./routes/query');

const app = express();
const PORT = process.env.PORT || 3001;

// 安全性中間件
app.use(helmet());

// CORS 設定
app.use(cors({
  origin: process.env.NODE_ENV === 'production' 
    ? ['https://your-domain.com'] 
    : ['http://localhost:3000'],
  credentials: true
}));

// 速率限制
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 分鐘
  max: 100, // 限制每個 IP 15 分鐘內最多 100 個請求
  message: {
    error: '請求過於頻繁，請稍後再試'
  }
});
app.use('/api/', limiter);

// 解析 JSON
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// 健康檢查端點
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    service: 'Medical Info Query System'
  });
});

// API 路由
app.use('/api/query', queryRoutes);

// 404 處理
app.use('*', (req, res) => {
  res.status(404).json({ error: '端點不存在' });
});

// 錯誤處理中間件
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ 
    error: '伺服器內部錯誤',
    message: process.env.NODE_ENV === 'development' ? err.message : '請稍後再試'
  });
});

app.listen(PORT, () => {
  console.log(`🚀 醫療資訊查詢系統後端運行在 http://localhost:${PORT}`);
  console.log(`📊 健康檢查: http://localhost:${PORT}/health`);
}); 