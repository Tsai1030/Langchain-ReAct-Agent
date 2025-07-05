# ReAct Medical Agent - 醫療資訊查詢系統

## 專案概述

本專案實現了一個基於 **ReAct (Reasoning and Acting)** 框架的醫療資訊查詢系統。該系統結合了本地醫療資料庫檢索（RAG）和網路即時搜尋功能，能夠智能地分析用戶查詢並提供準確的醫療資訊回應。

### 核心特色
- 🧠 **ReAct 框架實現**: 完整的 Thought → Action → Observation → Reasoning 循環
- 🔍 **多源資料整合**: 結合本地醫療資料庫與網路搜尋
- 🎯 **智能查詢分析**: 自動識別查詢類型和醫療急迫性
- 📊 **詳細步驟追蹤**: 完整記錄每一步的推理過程
- 🚀 **高效能 API**: 基於 FastAPI 的 RESTful 服務

## 系統架構

### 1. 整體架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                     ReAct Medical Agent                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Query Analyzer │  │  ReAct Engine   │  │  Response       │ │
│  │  - Intent       │  │  - Thought      │  │  - Formatter    │ │
│  │  - Urgency      │  │  - Action       │  │  - Validator    │ │
│  │  - Context      │  │  - Observation  │  │  - Synthesizer  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Action Toolkit                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  RAG Search     │  │  Web Search     │  │  Information    │ │
│  │  - Vector DB    │  │  - Serper API   │  │  - Validation   │ │
│  │  - Embedding    │  │  - Real-time    │  │  - Synthesis    │ │
│  │  - Retrieval    │  │  - Query Opt    │  │  - Formatting   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     FastAPI Service                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  REST API       │  │  Health Check   │  │  Logging &      │ │
│  │  - /api/ask     │  │  - Status       │  │  - Monitoring   │ │
│  │  - Error Handle │  │  - Metrics      │  │  - Tracing      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2. ReAct 框架實現

#### 核心循環流程
```python
for step in range(1, max_iterations + 1):
    # 1. THOUGHT - 分析當前狀況
    thought = self._thought_process(query, context, previous_steps)
    
    # 2. ACTION - 選擇最適合的動作
    action, action_input = self._select_action(query, context, previous_steps)
    
    # 3. OBSERVATION - 執行動作並觀察結果
    observation = await self._execute_action(action, action_input, context)
    
    # 4. REASONING - 基於觀察結果進行推理
    reasoning = self._create_reasoning(observation, action)
    
    # 5. FINISH - 判斷是否完成
    if action == ActionType.FINAL_ANSWER:
        break
```

#### 可用動作類型
| 動作類型 | 功能描述 | 使用場景 |
|---------|----------|----------|
| `ANALYZE_QUERY` | 深度分析用戶查詢意圖 | 每次查詢的起始步驟 |
| `SEARCH_RAG` | 搜尋本地醫療資料庫 | 醫師資訊、醫療知識查詢 |
| `SEARCH_WEB` | 網路即時搜尋 | 最新醫療資訊、時效性查詢 |
| `VALIDATE_INFORMATION` | 驗證資訊準確性 | 多源資料一致性檢查 |
| `FORMAT_RESPONSE` | 格式化回應內容 | 提供用戶友好的回應 |
| `FINAL_ANSWER` | 提供最終答案 | 完成查詢流程 |

## 核心模組設計

### 1. MedicalQueryAnalyzer - 查詢分析器
```python
class MedicalQueryAnalyzer:
    def analyze_query(self, query: str) -> QueryContext:
        # 智能分析查詢類型、急迫性、所需工具
        - 醫師資訊查詢 (doctor_info)
        - 醫療知識查詢 (medical_knowledge)
        - 一般查詢 (general)
        - 急迫性評估 (low/medium/high)
        - 時效性需求 (latest_info)
```

### 2. ReActMedicalAgent - 主要代理
```python
class ReActMedicalAgent:
    # 核心方法
    - query(): 主要查詢入口
    - _thought_process(): 思考過程
    - _select_action(): 動作選擇
    - _execute_action(): 動作執行
    - _create_reasoning(): 推理生成
```

### 3. 工具整合
- **RAG 系統**: Chroma 向量資料庫 + BGE-M3 嵌入模型
- **網路搜尋**: Google Serper API
- **語言模型**: GPT-4O (OpenAI)
- **向量檢索**: 語義相似性搜尋

## 技術實現細節

### 1. 智能查詢路由
```python
def analyze_query(self, query: str) -> QueryContext:
    # 關鍵字匹配
    doctor_keywords = ['醫師', '醫生', '博士', '教授', '主任']
    medical_keywords = ['治療', '症狀', '診斷', '藥物', '手術']
    urgent_keywords = ['急診', '緊急', '立即', '危險']
    
    # 智能路由決策
    if 醫師查詢 + 需要最新資訊 → 'both' (RAG + Web)
    elif 醫師查詢 → 'rag' (本地資料庫)
    elif 需要最新資訊 → 'web_search' (網路搜尋)
    else → 'both' (綜合查詢)
```

### 2. 多步驟推理記錄
```python
@dataclass
class ReActStep:
    step_number: int
    thought: str          # 思考過程
    action: ActionType    # 執行動作
    action_input: str     # 動作輸入
    observation: str      # 觀察結果
    reasoning: str        # 推理分析
    timestamp: datetime   # 時間戳記
```

### 3. 動作執行引擎
每個動作都有對應的執行方法：
- `_analyze_query_action()`: 使用 LLM 深度分析查詢
- `_search_rag_action()`: 向量檢索 + LLM 總結
- `_search_web_action()`: 網路搜尋 + LLM 處理
- `_validate_information_action()`: 一致性驗證
- `_format_response_action()`: 自然語言格式化
- `_final_answer_action()`: 整合所有資訊

## API 規格

### 主要端點
```http
POST /api/ask
Content-Type: application/json

{
    "question": "高雄醫學大學有哪些心臟科醫師？"
}
```

### 回應格式
```json
{
    "result": "根據查詢結果...",
    "query_type": "doctor_info",
    "medical_urgency": "low",
    "react_steps": [
        {
            "step": 1,
            "thought": "這是醫師資訊查詢...",
            "action": "analyze_query",
            "action_input": "高雄醫學大學有哪些心臟科醫師？",
            "observation": "查詢分析完成...",
            "reasoning": "根據查詢分析結果...",
            "timestamp": "2025-01-01T12:00:00"
        }
    ],
    "total_steps": 6,
    "processing_time": 3.45,
    "success": true
}
```

## 系統特色

### 1. 完整的 ReAct 實現
- ✅ **Thought**: 每步都有詳細的思考過程
- ✅ **Action**: 6 種專門的動作類型
- ✅ **Observation**: 詳實記錄每個動作的結果
- ✅ **Reasoning**: 基於觀察結果的邏輯推理
- ✅ **Finish**: 明確的終止條件

### 2. 智能化特色
- 🎯 **上下文感知**: 根據查詢類型選擇最適合的工具
- 🧠 **多步驟推理**: 複雜查詢的分步處理
- 🔄 **動態調整**: 根據中間結果調整後續策略
- 📊 **完整追蹤**: 詳細記錄每步的決策過程

### 3. 效能優化
- ⚡ **並行處理**: 異步執行提高響應速度
- 💾 **智能快取**: 避免重複計算
- 🎛️ **動作限制**: 最大 10 步避免無限循環
- 📈 **效能監控**: 詳細的時間和步驟統計

## 使用範例

### 醫師查詢
```
輸入: "高雄醫學大學心臟科有哪些醫師？"
步驟追蹤:
1. Thought: 這是醫師資訊查詢 → Action: analyze_query
2. Thought: 需要搜尋本地資料庫 → Action: search_rag
3. Thought: 需要驗證資訊 → Action: validate_information
4. Thought: 格式化回應 → Action: format_response
5. Thought: 提供最終答案 → Action: final_answer
```

### 醫療知識查詢
```
輸入: "2024年最新的心臟病治療方法"
步驟追蹤:
1. Thought: 需要最新資訊 → Action: analyze_query
2. Thought: 先搜尋網路 → Action: search_web
3. Thought: 補充本地資料 → Action: search_rag
4. Thought: 驗證一致性 → Action: validate_information
5. Thought: 格式化回應 → Action: format_response
6. Thought: 提供最終答案 → Action: final_answer
```

## 技術堆疊

### 後端框架
- **FastAPI**: 高效能 web 框架
- **LangChain**: LLM 應用框架
- **Chroma**: 向量資料庫
- **Uvicorn**: ASGI 服務器

### AI/ML 模型
- **GPT-4O**: OpenAI 語言模型
- **BGE-M3**: 多語言嵌入模型
- **HuggingFace**: 模型載入與管理

### 資料與搜尋
- **Google Serper API**: 網路搜尋
- **向量檢索**: 語義相似性搜尋
- **醫療資料庫**: 本地醫師資訊

## 部署與運行

### 環境需求
```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key

# 運行服務
python react_agent_api_v3.py
```

### 服務地址
- **主服務**: http://localhost:8000
- **API 端點**: http://localhost:8000/api/ask
- **健康檢查**: http://localhost:8000/api/health
- **API 文檔**: http://localhost:8000/docs

## 監控與日誌

### 日誌記錄
- 📝 **詳細步驟**: 每個 ReAct 步驟都有完整記錄
- 🕐 **時間追蹤**: 精確的執行時間統計
- 📊 **效能指標**: 步驟數、處理時間、成功率
- 🔍 **除錯資訊**: 詳細的錯誤追蹤

### 健康監控
- 系統狀態檢查
- 各元件運行狀況
- 快取使用情況
- 效能指標統計

## 未來擴展

### 短期目標
- [ ] 實現 ReAct 追蹤記錄持久化
- [ ] 添加更多專業醫療動作類型
- [ ] 優化查詢分析的準確度
- [ ] 加入使用者偏好學習

### 長期規劃
- [ ] 支援多語言查詢
- [ ] 集成更多醫療資料源
- [ ] 實現即時對話功能
- [ ] 添加視覺化推理過程

---

## 結論

本專案成功實現了一個完整的 ReAct 框架醫療查詢系統，展現了以下核心能力：

1. **完整的 ReAct 實現**: 嚴格遵循 Thought → Action → Observation → Reasoning 循環
2. **智能化決策**: 根據查詢類型和上下文動態選擇最適合的動作
3. **多源資料整合**: 有效結合本地資料庫和網路搜尋
4. **詳細的可追蹤性**: 完整記錄每一步的推理過程
5. **高效能實現**: 優化的異步處理和快取機制

系統具備了產品級的穩定性和擴展性，能夠處理複雜的醫療查詢並提供準確、可靠的回應。透過 ReAct 框架，系統展現了明確的推理能力和決策透明度，為醫療資訊查詢提供了可信賴的 AI 助手解決方案。
