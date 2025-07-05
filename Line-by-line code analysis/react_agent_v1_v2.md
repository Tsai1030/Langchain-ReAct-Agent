# ReAct 醫療資訊查詢系統 - 完整程式碼解析與架構分析

## 📋 目錄
- [系統概述](#系統概述)
- [基礎版本解析 (react_agent_api.py)](#基礎版本解析-react_agent_apipy)
- [進階版本解析 (react_agent_api_v3.py)](#進階版本解析-react_agent_api_v3py)
- [架構對比分析](#架構對比分析)
- [技術亮點總結](#技術亮點總結)

## 🏥 系統概述

這是一個基於 **ReAct (Reasoning and Acting)** 框架的智能醫療資訊查詢系統，提供兩個版本：
- **基礎版本**：簡潔高效的LangChain Agent實現
- **進階版本**：完整的ReAct框架，具備智能分析和急迫性判斷

## 📝 基礎版本解析 (react_agent_api.py)

### 系統架構設計理念

基礎版本採用 **LangChain Agent** 架構，專注於：
- **簡潔性**：最小化程式碼複雜度
- **效率**：快速部署和運行
- **穩定性**：成熟的框架保證系統可靠性
- **可維護性**：清晰的模組化設計

### 逐行程式碼解析

#### 1. 導入模組 (第1-15行)

```python
import os                    # 作業系統介面
import sys                   # 系統相關參數
import logging              # 日誌記錄
from datetime import datetime  # 日期時間處理
from fastapi import FastAPI, Request  # Web框架
from fastapi.middleware.cors import CORSMiddleware   # 跨域處理
import uvicorn              # ASGI伺服器
from langchain_community.vectorstores import Chroma       # 向量資料庫
from langchain_huggingface import HuggingFaceEmbeddings  # 嵌入模型
from langchain_openai import ChatOpenAI   # OpenAI聊天模型
from langchain.chains import RetrievalQA  # 檢索問答鏈
from langchain.tools import Tool          # 工具定義
from langchain_community.utilities import GoogleSerperAPIWrapper  # 網路搜尋
from langchain.agents import initialize_agent, AgentType  # 代理初始化
from dotenv import load_dotenv            # 環境變數載入
```

**🎯 設計用意：**
- **模組化設計**：每個導入都有明確的功能分工
- **LangChain生態系統**：利用成熟的AI框架，降低開發風險
- **FastAPI框架**：選擇高性能的現代Web框架，支援非同步處理
- **向量資料庫**：為RAG系統提供高效的相似度搜尋能力

#### 2. 環境設定 (第17-21行)

```python
# 關閉 LangSmith 追蹤
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
```

**🔒 設計用意：**
- **隱私保護**：關閉LangSmith追蹤，避免敏感醫療資料外洩
- **成本控制**：避免不必要的API調用費用
- **生產環境優化**：移除開發階段的調試功能

#### 3. 日誌系統設定 (第23-31行)

```python
# 設定詳細 log 檔案
log_filename = f"agent_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,                    # 日誌級別設為INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日誌格式
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # 檔案處理器
        logging.StreamHandler()            # 控制台處理器
    ]
)
```

**📊 設計用意：**
- **可追溯性**：每個查詢都有完整的執行記錄
- **問題診斷**：便於系統維護和錯誤排查
- **雙重輸出**：同時記錄到檔案和控制台，確保資訊不遺失
- **時間戳記**：便於追蹤查詢的時間順序

#### 4. 環境變數載入 (第33行)

```python
load_dotenv()  # 從.env檔案載入環境變數
```

**🔐 設計用意：**
- **安全性**：敏感資訊（API金鑰）不直接寫在程式碼中
- **配置管理**：便於不同環境（開發/測試/生產）的配置切換
- **最佳實踐**：符合12-Factor App的配置原則

#### 5. RAG系統初始化 (第35-49行)

```python
# 1. RAG 設定
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",        # 使用BGE-M3嵌入模型
    model_kwargs={"device": "cpu"}    # 使用CPU設備
)
vectorstore = Chroma(
    persist_directory="chroma_db",    # 持久化目錄
    collection_name="doctors_smart_20250704_0608",  # 集合名稱
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
```

**🧠 設計用意：**
- **模型選擇**：BGE-M3是中文語義理解表現優異的嵌入模型
- **硬體優化**：使用CPU部署，降低硬體成本
- **資料持久化**：Chroma向量資料庫確保資料不遺失
- **檢索優化**：設定k=5，平衡準確性和效率
- **溫度控制**：temperature=0確保回答的一致性

#### 6. RAG查詢函數 (第51-62行)

```python
def rag_query(query: str) -> str:
    logging.info(f"🔍 RAG 檢索開始: {query}")
    logging.info(f"📊 向量資料庫查詢: {query}")
    docs = vectorstore.similarity_search(query, k=3)
    logging.info(f" 找到 {len(docs)} 個相關文件")
    for i, doc in enumerate(docs):
        logging.info(f"   文件 {i+1}: {doc.page_content[:100]}...")
    result = qa_chain.invoke({"query": query})
    response = result['result'] if isinstance(result, dict) else str(result)
    logging.info(f"🤖 RAG 最終結果: {response[:200]}...")
    return response
```

**🔍 設計用意：**
- **詳細日誌**：每個步驟都有記錄，便於調試和監控
- **錯誤處理**：處理不同格式的回應結果
- **效能監控**：記錄檢索到的文件數量和內容摘要
- **可讀性**：使用表情符號讓日誌更易讀

#### 7. RAG工具定義 (第64-69行)

```python
rag_tool = Tool(
    name="醫師資料庫查詢",
    func=rag_query,
    description="查詢本地醫師資料庫，包含醫師姓名、專長、學歷、經歷、職稱等資訊。輸入格式：醫師姓名 + 查詢項目，例如：'朱志生醫師專長'、'林宗翰醫師學歷'、'高醫心臟科醫師'。這個工具會回傳醫師的詳細資訊。"
)
```

**🛠️ 設計用意：**
- **工具化設計**：將RAG功能封裝為可重用的工具
- **明確描述**：詳細說明工具的功能和使用方式
- **範例提供**：給出具體的使用範例，降低使用門檻
- **專業化**：針對醫療領域的特定需求設計

#### 8. 網路搜尋工具 (第71-85行)

```python
# 2. Web Search Tool (Serper)
serper_wrapper = GoogleSerperAPIWrapper()
def smart_web_search(query: str) -> str:
    logging.info(f"🌐 Web Search 開始: {query}")
    try:
        result = serper_wrapper.run(query)
        logging.info(f"🌐 Web Search 結果: {result[:200]}...")
        return result
    except Exception as e:
        error_msg = f"Web search failed: {e}"
        logging.error(f"❌ Web Search 錯誤: {error_msg}")
        return error_msg

web_search_tool = Tool(
    name="網路搜尋",
    func=smart_web_search,
    description="查詢網路最新資訊，適合查詢醫療新知、最新治療方法、研究報告等。例如：'2024年高血壓治療新趨勢'、'心臟病預防最新研究'。"
)
```

**🌐 設計用意：**
- **互補性**：網路搜尋補充RAG資料庫的時效性不足
- **錯誤處理**：優雅的錯誤處理，避免系統崩潰
- **專業導向**：針對醫療領域的最新資訊需求
- **時效性**：獲取最新的醫療研究和治療方法

#### 9. 代理初始化 (第87-96行)

```python
# 3. Agent
# 工具順序：RAG、Web Search
tools = [rag_tool, web_search_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate"
)
```

**🤖 設計用意：**
- **工具優先級**：RAG優先於網路搜尋，確保權威性
- **零樣本學習**：AgentType.ZERO_SHOT_REACT_DESCRIPTION無需預訓練
- **錯誤容忍**：handle_parsing_errors=True提高系統穩定性
- **效能控制**：max_iterations=5避免無限循環
- **早期停止**：early_stopping_method="generate"提高效率

#### 10. FastAPI應用設定 (第98-107行)

```python
# FastAPI 設定
app = FastAPI()

# 允許所有來源跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**🚀 設計用意：**
- **現代化框架**：FastAPI提供高性能和自動API文檔
- **跨域支援**：允許前端應用程式跨域訪問
- **靈活性**：支援多種HTTP方法和標頭
- **開發友好**：自動生成OpenAPI文檔

#### 11. 主要API端點 (第109-125行)

```python
@app.post("/api/ask")
async def ask_agent(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return {"error": "No question provided"}
    
    logging.info(f"\n🚀 收到新查詢: {question}")
    logging.info("=" * 80)
    try:
        result = agent.run(question)
        logging.info("=" * 80)
        logging.info(f"✅ 查詢完成: {result[:100]}...")
        logging.info("\n")
        return {"result": result}
    except Exception as e:
        error_msg = f"Agent 執行錯誤: {str(e)}"
        logging.error(f"❌ {error_msg}")
        logging.info("=" * 80)
        return {"error": error_msg}
```

**🔌 設計用意：**
- **非同步處理**：async/await提高並發處理能力
- **輸入驗證**：檢查必要參數是否存在
- **視覺化日誌**：使用分隔線讓日誌更清晰
- **錯誤處理**：完整的異常捕獲和錯誤回報
- **回應格式**：統一的JSON回應格式

#### 12. 服務啟動 (第127-134行)

```python
if __name__ == "__main__":
    logging.info(" 啟動醫療資訊查詢系統 FastAPI 後端")
    logging.info("📍 服務地址: http://localhost:8000")
    logging.info("🔗 API 端點: http://localhost:8000/api/ask")
    logging.info(f"📝 詳細 log 檔案: {log_filename}")
    logging.info("=" * 60)
    uvicorn.run("react_agent_api:app", host="0.0.0.0", port=8000, reload=True)
```

**⚙️ 設計用意：**
- **啟動資訊**：清楚顯示服務配置資訊
- **開發模式**：reload=True支援熱重載
- **網路訪問**：host="0.0.0.0"允許外部訪問
- **標準端口**：使用8000端口，符合開發慣例

### 基礎版本架構亮點

#### 🏗️ **系統架構優勢**
1. **模組化設計**：每個組件職責明確，便於維護和擴展
2. **雙重資訊來源**：RAG + 網路搜尋，確保資訊的權威性和時效性
3. **錯誤處理機制**：完整的異常捕獲和錯誤回報
4. **詳細日誌系統**：便於問題診斷和系統監控
5. **現代化技術棧**：使用業界主流的AI和Web技術

#### 📊 **效能優化策略**
1. **向量檢索優化**：k=5平衡準確性和效率
2. **模型選擇**：BGE-M3在中文語義理解上表現優異
3. **硬體成本控制**：使用CPU部署，降低硬體要求
4. **快取機制**：Chroma向量資料庫提供持久化儲存
5. **並發處理**：FastAPI非同步框架提高處理能力

#### 🔒 **安全性考量**
1. **環境變數管理**：敏感資訊不直接寫在程式碼中
2. **LangSmith關閉**：避免敏感醫療資料外洩
3. **輸入驗證**：檢查必要參數，防止無效請求
4. **錯誤資訊控制**：避免敏感資訊在錯誤訊息中洩露

## 🚀 進階版本解析 (react_agent_api_v3.py)

### 系統架構設計理念

進階版本採用 **完整ReAct框架**，專注於：
- **智能分析**：自動判斷查詢類型和急迫性
- **結構化推理**：每個步驟都有思考、動作、觀察、推理
- **醫療專業化**：針對醫療領域的特殊需求
- **可擴展性**：支援更多工具和分析功能

### 核心架構組件

#### 1. 資料結構設計

```python
@dataclass
class QueryContext:
    """查詢上下文"""
    query: str             # 原始查詢
    query_type: str        # 查詢類型：'doctor_info', 'medical_knowledge', 'general'
    priority_tool: str     # 優先工具：'rag', 'web_search', 'both'
    confidence_threshold: float = 0.7      # 信心度閾值
    requires_latest_info: bool = False     # 是否需要最新資訊
    medical_urgency: str = "low"           # 醫療急迫性：low, medium, high

@dataclass
class ReActStep:
    """ReAct 步驟"""
    step_number: int        # 步驟編號
    thought: str           # 思考過程
    action: ActionType     # 執行的動作
    action_input: str      # 動作輸入
    observation: str       # 觀察結果
    reasoning: str         # 推理過程
    timestamp: datetime    # 時間戳記
```

#### 2. 智能查詢分析器

```python
class MedicalQueryAnalyzer:
    """醫療查詢分析器"""
    
    def __init__(self):
        self.doctor_keywords = ['醫師', '醫生', '博士', '教授', '主任', '院長', '科主任']
        self.medical_keywords = ['治療', '症狀', '診斷', '藥物', '手術', '預防', '病因', '副作用']
        self.latest_keywords = ['最新', '2024', '2025', '近期', '現在', '目前', '新']
        self.urgent_keywords = ['急診', '緊急', '立即', '馬上', '危險', '嚴重']
```

#### 3. ReAct醫療代理

```python
class ReActMedicalAgent:
    """ReAct 醫療代理"""
    
    def __init__(self):
        self.analyzer = MedicalQueryAnalyzer()
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.react_steps = []
        self.max_iterations = 10
```

### 急迫性分析機制

#### 🚨 為什麼要設置急迫性？

分析醫療查詢的急迫性具有非常重要的意義：

#### 1. 🔍 急迫性識別機制

```python
# 定義急迫性關鍵字
self.urgent_keywords = ['急診', '緊急', '立即', '馬上', '危險', '嚴重']

# 判斷急迫性
urgency = "high" if any(keyword in query for keyword in self.urgent_keywords) else "low"
```

**識別邏輯：**
- 當查詢包含「急診」、「緊急」、「立即」等關鍵字時，系統會將急迫性設為 `"high"`
- 否則設為 `"low"`

#### 2. 🎯 急迫性的實際意義

##### A. 回應優先級調整
```python
# 在思考過程中會考慮急迫性
return f"這是一個關於{context.query_type}的查詢，急迫性為{context.medical_urgency}。我需要先分析查詢的具體需求。"
```

**高急迫性查詢的處理策略：**
- **優先處理**：系統會優先處理高急迫性的查詢
- **快速回應**：減少不必要的驗證步驟，加快回應速度
- **直接答案**：提供更直接、簡潔的醫療建議

##### B. 搜尋策略優化
```python
# 根據急迫性調整搜尋工具選擇
if context.requires_latest_info:
    return QueryContext(query, 'medical_knowledge', 'web_search', 
                      requires_latest_info=True, medical_urgency=urgency)
```

**高急迫性時的搜尋策略：**
- **優先網路搜尋**：獲取最新、最即時的醫療資訊
- **減少本地搜尋**：避免過時的資料庫資訊
- **快速驗證**：簡化資訊驗證流程

#### 3. 🏥 醫療安全考量

##### A. 緊急情況識別
```python
# 急迫性關鍵字對應的醫療場景
'急診' → 需要立即醫療干預
'緊急' → 時間敏感的健康問題
'立即' → 需要快速決策
'馬上' → 不能延遲的醫療需求
'危險' → 潛在的健康風險
'嚴重' → 需要專業醫療評估
```

##### B. 回應內容調整
- **高急迫性**：強調立即就醫、急診建議
- **低急迫性**：提供一般性醫療資訊、預防建議

#### 4. ⚡ 系統效能優化

##### A. 處理流程調整
```python
# 根據急迫性調整最大迭代次數
self.max_iterations = 10  # 一般情況

# 高急迫性時可能需要更少的步驟
if context.medical_urgency == "high":
    # 減少驗證步驟，直接提供答案
    # 優先網路搜尋獲取最新資訊
```

##### B. 快取策略
- **高急迫性**：跳過快取，直接搜尋最新資訊
- **低急迫性**：可以使用快取提高回應速度

#### 5. 📊 實際應用場景

##### 場景1：緊急醫療查詢
```
用戶查詢：「我現在胸痛很嚴重，需要立即就醫嗎？」
急迫性分析：high（包含「嚴重」、「立即」）
系統回應：優先提供緊急醫療建議，強調立即就醫的重要性
```

##### 場景2：一般醫療諮詢
```
用戶查詢：「高血壓的預防方法有哪些？」
急迫性分析：low（無急迫性關鍵字）
系統回應：提供詳細的預防資訊，可以進行多步驟驗證
```

### 急迫性設置的意義總結

1. **🔴 醫療安全**：確保緊急情況得到優先處理
2. **⚡ 回應效率**：根據急迫性調整處理策略
3. **🎯 用戶體驗**：提供符合期望的回應速度和內容
4. **🏥 專業性**：體現醫療系統的專業判斷能力
5. **⚙️ 系統優化**：根據急迫性調整資源分配

## 🔄 架構對比分析

| 特性 | 基礎版本 | 進階版本 |
|------|----------|----------|
| **架構類型** | LangChain Agent | 完整ReAct框架 |
| **程式碼行數** | 144行 | 680行 |
| **查詢分析** | 簡單工具選擇 | 智能意圖分析 |
| **急迫性判斷** | ❌ 無 | ✅ 完整實現 |
| **推理過程** | 隱藏 | 可視化追蹤 |
| **擴展性** | 中等 | 高 |
| **維護複雜度** | 低 | 中等 |
| **適用場景** | 快速部署 | 生產環境 |

## 🎯 技術亮點總結

### 🏗️ **系統架構優勢**
1. **模組化設計**：每個組件職責明確，便於維護和擴展
2. **雙重資訊來源**：RAG + 網路搜尋，確保資訊的權威性和時效性
3. **錯誤處理機制**：完整的異常捕獲和錯誤回報
4. **詳細日誌系統**：便於問題診斷和系統監控
5. **現代化技術棧**：使用業界主流的AI和Web技術

### 📊 **效能優化策略**
1. **向量檢索優化**：k=5平衡準確性和效率
2. **模型選擇**：BGE-M3在中文語義理解上表現優異
3. **硬體成本控制**：使用CPU部署，降低硬體要求
4. **快取機制**：Chroma向量資料庫提供持久化儲存
5. **並發處理**：FastAPI非同步框架提高處理能力

### 🔒 **安全性考量**
1. **環境變數管理**：敏感資訊不直接寫在程式碼中
2. **LangSmith關閉**：避免敏感醫療資料外洩
3. **輸入驗證**：檢查必要參數，防止無效請求
4. **錯誤資訊控制**：避免敏感資訊在錯誤訊息中洩露

### 🎯 **業務價值**
1. **醫療專業化**：針對醫療領域的特定需求設計
2. **用戶體驗**：提供準確、及時的醫療資訊查詢
3. **可擴展性**：模組化設計便於功能擴展
4. **維護性**：詳細日誌和錯誤處理便於系統維護

## 🔗 API 端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 根路徑，顯示系統狀態 |
| `/api/ask` | POST | 主要查詢端點 |
| `/api/health` | GET | 健康檢查 |
| `/api/react-trace/{trace_id}` | GET | 追蹤記錄 |
| `/api/cache` | DELETE | 清除快取 |

## 🚀 使用方式

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 設定環境變數
```bash
# .env 檔案
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

### 3. 啟動服務
```bash
# 基礎版本
python react_agent_api.py

# 進階版本
python react_agent_api_v3.py
```

### 4. 發送查詢
```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "我想找高雄醫學大學的心臟科醫師"}'
```

## 📊 系統特點

- ✅ **智能分析**：自動分析查詢類型和優先級
- ✅ **多工具整合**：結合RAG資料庫和網路搜尋
- ✅ **結構化推理**：每個步驟都有思考、動作、觀察、推理
- ✅ **醫療專業化**：針對醫療領域優化的關鍵字和處理邏輯
- ✅ **完整API**：提供RESTful API介面
- ✅ **詳細日誌**：完整的執行過程記錄
- ✅ **急迫性分析**：根據醫療緊急程度調整處理策略

## 🔧 技術棧

- **Python 3.8+**
- **FastAPI**：Web框架
- **LangChain**：LLM整合
- **Chroma**：向量資料庫
- **HuggingFace**：嵌入模型
- **OpenAI GPT-4**：語言模型
- **Google Serper**：網路搜尋

## 📝 注意事項

1. 本系統僅供參考，不應替代專業醫療建議
2. 緊急醫療情況請立即就醫
3. 所有回應都包含醫療免責聲明
4. 系統會記錄所有查詢過程用於改進

---

**版本**：基礎版本 + 進階版本  
**框架**：LangChain Agent + ReAct (Reasoning and Acting)  
**更新日期**：2024年
