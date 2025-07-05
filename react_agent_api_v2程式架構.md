
# 醫療查詢系統架構深度解析

## 🏗️ 整體架構圖

```
用戶請求 → FastAPI → 查詢分析器 → 策略選擇器 → 執行引擎 → 格式化器 → 回應
    ↓           ↓           ↓            ↓           ↓           ↓
   HTTP      路由處理    意圖識別      工具選擇    並行查詢    專業格式
```

## 📦 核心組件架構

### 1. 數據容器層 (Data Container Layer)

#### QueryContext 數據類
```python
@dataclass
class QueryContext:
    query: str              # 原始查詢
    query_type: str         # 查詢類型分類
    priority_tool: str      # 優先工具選擇
    confidence_threshold: float = 0.7
```

**設計用意**：
- 封裝查詢上下文信息
- 類型安全的數據傳遞
- 為後續決策提供結構化數據

**實現方式**：
- 使用 `@dataclass` 自動生成構造函數
- 提供默認值支持
- 類型註解確保數據一致性

### 2. 智能分析層 (Intelligence Analysis Layer)

#### MedicalQueryAnalyzer - 意圖識別引擎
```python
class MedicalQueryAnalyzer:
    def __init__(self):
        # 關鍵字字典建立語義映射
        self.doctor_keywords = ['醫師', '醫生', '博士', '教授']
        self.medical_keywords = ['治療', '症狀', '診斷', '藥物']
        self.latest_keywords = ['最新', '2024', '2025', '近期']
```

**核心算法**：
```python
def analyze_query(self, query: str) -> QueryContext:
    # 1. 醫師查詢檢測
    if any(keyword in query for keyword in self.doctor_keywords):
        if any(keyword in query for keyword in self.latest_keywords):
            return QueryContext(query, 'doctor_info', 'both')  # 需要最新資訊
        return QueryContext(query, 'doctor_info', 'rag')       # 本地資料庫足夠
    
    # 2. 最新資訊檢測
    if any(keyword in query for keyword in self.latest_keywords):
        return QueryContext(query, 'medical_knowledge', 'web_search')
    
    # 3. 醫療知識檢測
    if any(keyword in query for keyword in self.medical_keywords):
        return QueryContext(query, 'medical_knowledge', 'both')
```

**設計用意**：
- **語義理解**：將自然語言映射到結構化意圖
- **策略決策**：根據查詢特徵選擇最優工具組合
- **效率優化**：避免不必要的資源消耗

### 3. 執行引擎層 (Execution Engine Layer)

#### ImprovedMedicalAgent - 核心查詢引擎

##### 初始化架構
```python
def __init__(self):
    self.analyzer = MedicalQueryAnalyzer()        # 分析器
    self.formatter = MedicalResponseFormatter()   # 格式化器
    self.cache = {}                               # 快取系統
    self.cache_ttl = timedelta(hours=1)          # 快取生命週期
    
    # 工具鏈初始化
    self._init_rag_system()      # RAG 系統
    self._init_web_search()      # 網路搜尋
    self._create_custom_prompts() # 提示模板
    self._create_query_chains()   # 查詢鏈
```

**設計用意**：
- **模組化設計**：每個功能獨立初始化
- **故障隔離**：某個模組失效不影響其他模組
- **資源管理**：統一管理所有外部資源

##### RAG 系統架構
```python
def _init_rag_system(self):
    # 1. 嵌入模型初始化
    self.embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",           # 多語言嵌入模型
        model_kwargs={"device": "cpu"}      # CPU 運行確保穩定性
    )
    
    # 2. 向量資料庫連接
    self.vectorstore = Chroma(
        persist_directory="chroma_db",              # 持久化目錄
        collection_name="doctors_smart_20250704",   # 集合名稱
        embedding_function=self.embedding_model     # 嵌入函數
    )
    
    # 3. 檢索器配置
    self.retriever = self.vectorstore.as_retriever(
        search_kwargs={"k": 5}  # 返回前5個最相關文檔
    )
```

**實現原理**：
- **語義搜尋**：使用 BGE-M3 模型將查詢轉換為向量
- **相似度計算**：在向量空間中找到最相似的文檔
- **結果排序**：根據相似度分數排序返回

##### 並行查詢架構
```python
async def _integrated_query(self, question: str) -> str:
    # 1. 創建並行任務
    rag_task = asyncio.create_task(self._rag_only_query(question))
    web_task = asyncio.create_task(self._web_only_query(question))
    
    # 2. 等待所有任務完成
    rag_result, web_result = await asyncio.gather(rag_task, web_task)
    
    # 3. 結果整合
    integration_prompt = self.integration_prompt.format(
        query=question,
        rag_result=rag_result,
        web_result=web_result
    )
    
    response = await asyncio.to_thread(self.llm.invoke, integration_prompt)
    return response.content
```

**設計優勢**：
- **時間優化**：並行執行減少總等待時間
- **資源利用**：充分利用多核心處理能力
- **故障恢復**：單個任務失敗不影響整體結果

### 4. 快取系統架構

#### 智能快取機制
```python
def _rag_query(self, query: str) -> str:
    # 1. 快取鍵生成
    cache_key = f"rag_{hash(query)}"
    
    # 2. 快取檢查
    if cache_key in self.cache:
        cached_result, timestamp = self.cache[cache_key]
        if datetime.now() - timestamp < self.cache_ttl:
            return cached_result  # 返回快取結果
    
    # 3. 執行查詢
    result = self.rag_chain.invoke(query)
    
    # 4. 更新快取
    self.cache[cache_key] = (result, datetime.now())
    return result
```

**快取策略**：
- **TTL 機制**：1小時後自動失效
- **哈希鍵**：確保相同查詢使用相同快取
- **時間戳**：記錄快取創建時間

### 5. 提示工程架構

#### 專業化提示模板
```python
self.doctor_info_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
你是一位專業的醫療資訊助手。請基於以下資料庫內容，回答用戶關於醫師的問題。

用戶問題：{query}
相關資料：{context}

請按以下格式回答：
【醫師資訊】
• 醫師姓名：
• 專長領域：
• 學歷背景：
• 職務經歷：
• 聯絡資訊：

回答要求：
- 資訊要準確完整
- 如果資料不足，請明確說明
- 使用繁體中文回答
- 保持專業醫療用語
"""
)
```

**設計理念**：
- **角色設定**：明確 AI 助手的專業身份
- **格式規範**：統一的輸出格式
- **品質要求**：明確的準確性和完整性要求

### 6. 查詢鏈架構

#### LangChain 鏈式處理
```python
def _create_query_chains(self):
    # RAG 查詢鏈
    self.rag_chain = (
        {
            "context": self.retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
            "query": RunnablePassthrough()
        }
        | self.doctor_info_prompt    # 提示模板
        | self.llm                   # 語言模型
        | StrOutputParser()          # 輸出解析器
    )
```

**處理流程**：
1. **數據檢索**：`self.retriever` 從向量資料庫檢索相關文檔
2. **上下文構建**：將檢索到的文檔內容合併為上下文
3. **提示生成**：將查詢和上下文填入提示模板
4. **模型推理**：LLM 基於提示生成回答
5. **輸出解析**：提取最終的字符串結果

### 7. API 服務層架構

#### FastAPI 路由設計
```python
@app.post("/api/ask")
async def ask_medical_agent(request: Request):
    # 1. 參數驗證
    data = await request.json()
    question = data.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="問題不能為空")
    
    # 2. 查詢處理
    result = await medical_agent.query(question)
    
    # 3. 結果返回
    return result
```

**架構特點**：
- **異步處理**：支持並發請求
- **錯誤處理**：統一的異常處理機制
- **參數驗證**：確保輸入數據有效性

## 🔄 數據流程圖

```
用戶查詢 "李醫師的專長是什麼？"
    ↓
QueryAnalyzer 分析
    ↓
檢測到 "醫師" 關鍵字 → query_type: "doctor_info", priority_tool: "rag"
    ↓
執行 RAG 查詢
    ↓
向量檢索 → 提示生成 → LLM 推理 → 結果解析
    ↓
MedicalResponseFormatter 格式化
    ↓
返回結構化醫師資訊
```

## 🚀 性能優化策略

### 1. 快取優化
- **查詢快取**：避免重複檢索
- **TTL 機制**：平衡資料新鮮度和性能

### 2. 並行處理
- **異步任務**：同時執行多個查詢
- **資源隔離**：避免任務間相互影響

### 3. 故障恢復
- **降級策略**：Web 搜尋失敗時仍能提供 RAG 結果
- **錯誤隔離**：單個組件故障不影響整體系統

## 🎯 架構優勢總結

1. **模組化設計**：各組件獨立，易於維護和擴展
2. **智能路由**：根據查詢特徵選擇最優處理策略
3. **性能優化**：多層快取和並行處理
4. **故障恢復**：完善的降級和錯誤處理機制
5. **專業化**：針對醫療領域的定制化設計

這個架構展現了從簡單工具使用到複雜系統設計的完整思路，每個組件都有明確的職責和優化目標。
