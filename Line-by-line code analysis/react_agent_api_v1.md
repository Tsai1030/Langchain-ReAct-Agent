# 醫療資訊智能查詢系統 - 技術架構說明

## 📋 專案概述

本系統是一個整合式醫療資訊查詢平台，結合了本地醫師資料庫檢索 (RAG) 和即時網路搜尋功能，為使用者提供全面且準確的醫療資訊服務。

### 🎯 核心價值主張
- **雙重資訊來源**：結合本地權威醫師資料與最新網路醫療資訊
- **智能檢索**：運用向量資料庫 (Vector Database) 實現語意搜尋
- **即時回應**：透過 AI Agent 架構提供智能化的查詢處理
- **企業級架構**：採用微服務設計，具備高可用性和擴展性

## 🏗️ 系統架構

### 整體架構圖
```
┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐
│   前端介面       │    │   FastAPI       │     │   資料儲存層     │
│   (React)       │◄──►│   後端服務       │◄──► │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘     └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AI Agent      │
                       │   (LangChain)   │
                       └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌─────────────┐     ┌─────────────┐
                RAG工具           網路搜尋工具 
               (醫師資料)         (Serper API)
            └─────────────┘     └─────────────┘
```

## 🔧 技術棧 (Technology Stack)

### 後端技術 (Backend)
- **FastAPI**: 現代化 Python Web 框架，提供高性能 API 服務
- **LangChain**: AI 應用開發框架，實現智能 Agent 邏輯
- **ChromaDB**: 向量資料庫，支援語意搜尋功能
- **HuggingFace Embeddings**: 使用 BAAI/bge-m3 模型進行文本向量化
- **OpenAI GPT-4o**: 大語言模型，提供自然語言理解與生成

### 前端技術 (Frontend)
- **React 18**: 現代化前端框架
- **TypeScript**: 型別安全的 JavaScript 超集
- **Tailwind CSS**: 實用優先的 CSS 框架
- **Vite**: 快速建置工具
- **Axios**: HTTP 客戶端庫

### 資料處理 (Data Processing)
- **Python**: 主要開發語言
- **Uvicorn**: ASGI 伺服器
- **Python-dotenv**: 環境變數管理

## 🧠 核心功能模組

### 1. RAG (Retrieval-Augmented Generation) 模組
```python
# 向量資料庫配置
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="doctors_smart_20250704_0608",
    embedding_function=embedding_model
)
```

**功能特色：**
- 使用本地醫師資料庫進行語意搜尋
- 支援醫師姓名、專長、學歷等多維度查詢
- 檢索相關性排序，確保結果準確性

### 2. 網路搜尋模組
```python
serper_wrapper = GoogleSerperAPIWrapper()
def smart_web_search(query: str) -> str:
    # 即時網路搜尋最新醫療資訊
```

**功能特色：**
- 整合 Google Serper API 進行即時網路搜尋
- 獲取最新醫療研究、治療方法等動態資訊
- 與本地資料互補，提供完整資訊覆蓋

### 3. AI Agent 智能調度
```python
agent = initialize_agent(
    tools=[rag_tool, web_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate"
)
```

**功能特色：**
- 智能判斷查詢類型，自動選擇合適工具
- 支援多輪對話和複雜查詢處理
- 具備錯誤處理和重試機制

## 📊 資料架構

### 向量資料庫設計
- **儲存格式**: ChromaDB 向量資料庫
- **嵌入模型**: BAAI/bge-m3 (多語言支援)
- **檢索策略**: 相似度搜尋 (Similarity Search)
- **檢索數量**: 預設檢索前 5 個最相關文件

### 資料處理流程
1. **資料預處理**: 醫師資訊結構化處理
2. **向量化**: 使用嵌入模型轉換為向量表示
3. **索引建立**: 建立向量索引以加速檢索
4. **持久化儲存**: 資料持久化到 ChromaDB

## 🔄 API 設計

### 主要端點
```http
POST /api/ask
Content-Type: application/json

{
  "question": "查詢內容"
}
```

### 回應格式
```json
{
  "result": "AI 處理後的回答",
  "error": "錯誤訊息 (如有)"
}
```

## 🛡️ 系統安全與監控

### 日誌系統
- **結構化日誌**: 詳細記錄系統運行狀態
- **時間戳記**: 每個操作都有精確時間記錄
- **錯誤追蹤**: 完整的錯誤堆疊追蹤
- **效能監控**: 查詢響應時間統計

### 安全措施
- **環境變數**: 敏感資訊透過 .env 檔案管理
- **CORS 配置**: 跨域請求安全控制
- **輸入驗證**: API 輸入參數驗證
- **錯誤處理**: 優雅的錯誤處理機制

## 📈 效能優化

### 檢索優化
- **向量索引**: 使用 ChromaDB 高效向量索引
- **檢索數量控制**: 限制檢索文件數量以提升速度
- **快取機制**: 支援查詢結果快取

### 系統效能
- **非同步處理**: 使用 FastAPI 非同步特性
- **記憶體管理**: 優化的記憶體使用策略
- **並發處理**: 支援多用戶同時查詢

## 🚀 部署架構

### 開發環境
```bash
# 後端啟動
python react_agent_api.py

# 前端啟動
cd client && npm run dev
```

### 生產環境建議
- **容器化**: 使用 Docker 進行容器化部署
- **負載平衡**: 配置 Nginx 負載平衡器
- **監控**: 整合 Prometheus + Grafana 監控
- **備份**: 定期備份向量資料庫

## 💼 商業價值

### 目標用戶
- **醫療機構**: 醫師資訊管理與查詢
- **患者**: 醫師背景與專長查詢
- **研究人員**: 醫療資訊研究與分析

### 競爭優勢
1. **資訊完整性**: 結合本地權威資料與最新網路資訊
2. **查詢智能性**: AI 驅動的智能檢索與回答
3. **系統可擴展性**: 模組化設計，易於功能擴展
4. **技術先進性**: 採用最新的 AI 和向量資料庫技術

### 未來發展方向
- **多語言支援**: 擴展至多語言查詢
- **語音介面**: 整合語音識別與合成
- **個人化推薦**: 基於用戶行為的個性化推薦
- **醫療影像分析**: 整合醫療影像 AI 分析功能

## 📋 技術債務與改進建議

### 短期改進
- [ ] 增加 API 速率限制
- [ ] 實作查詢結果快取機制
- [ ] 優化向量檢索演算法
- [ ] 增加單元測試覆蓋率

### 長期規劃
- [ ] 微服務架構重構
- [ ] 分散式向量資料庫部署
- [ ] 機器學習模型持續優化
- [ ] 多租戶架構支援

## 📞 技術支援

### 開發團隊
- **後端開發**: Python/FastAPI 專家
- **前端開發**: React/TypeScript 專家
- **AI/ML 工程師**: LangChain/向量資料庫專家
- **DevOps 工程師**: 部署與監控專家

### 文件資源
- **API 文件**: FastAPI 自動生成
- **技術文件**: 詳細的架構與部署指南
- **用戶手冊**: 系統使用說明

---

*本文件最後更新：2024年7月5日*
*版本：v1.0*
