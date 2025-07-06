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

| 層級 | 技術 | 用途 |
|------|------|------|
| **語言基礎** | Python 3.8+ | 主要開發語言 |
| **API 層** | FastAPI | 高效能非同步 API 服務 |
| **AI 層** | GPT-4O + LangChain | 自然語言理解與生成 |
| **檢索層** | ChromaDB + BGE-M3 | 向量資料庫與語義檢索 |
| **搜尋層** | Google Serper API | 即時網路資訊獲取 |
| **部署層** | Uvicorn | ASGI 伺服器 |
| **配置管理** | Python-dotenv | 環境變數管理 |

## 🧠 ReAct 框架實現

### 六大核心動作
1. **ANALYZE_QUERY** - 智能查詢分析
2. **SEARCH_RAG** - 本地資料庫檢索
3. **SEARCH_WEB** - 網路即時搜尋
4. **VALIDATE_INFORMATION** - 資訊交叉驗證
5. **FORMAT_RESPONSE** - 專業格式化
6. **FINAL_ANSWER** - 整合式回答

### 推理流程

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
```
用戶查詢 → 意圖分析 → 策略選擇 → 資訊檢索 → 交叉驗證 → 格式化 → 最終回答
    ↓         ↓         ↓         ↓         ↓         ↓         ↓
  Thought  Action  Observation  Reasoning  Action  Observation  Answer
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
```
用戶查詢 → Agent 分析 → 選擇工具 → 執行工具 → 處理結果 → 生成回答
    ↓
如果還需要更多資訊 → 繼續迭代 (最多5次)
    ↓
達到最大迭代次數或獲得滿意答案 → 停止並返回結果
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

---

*本文件最後更新：2024年7月5日*
*版本：v1.0*
