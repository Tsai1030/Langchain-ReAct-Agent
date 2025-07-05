# ReAct 醫療資訊查詢系統 - 完整程式碼解析

## 📋 目錄
- [系統概述](#系統概述)
- [逐行程式碼解析](#逐行程式碼解析)
- [急迫性分析機制](#急迫性分析機制)
- [系統架構](#系統架構)
- [API 端點](#api-端點)
- [使用方式](#使用方式)

## 🏥 系統概述

這是一個基於 **ReAct (Reasoning and Acting)** 框架的智能醫療資訊查詢系統，能夠：
- 自動分析醫療查詢類型和急迫性
- 結合 RAG 資料庫和網路搜尋
- 提供結構化的推理過程
- 支援多種醫療查詢類型

## 📝 逐行程式碼解析

### 1. 導入模組 (第1-25行)

```python
import os                    # 作業系統介面
import sys                   # 系統相關參數
import logging              # 日誌記錄
import asyncio              # 非同步程式設計
from datetime import datetime, timedelta  # 日期時間處理
from typing import Dict, List, Optional, Any, Tuple  # 型別提示
from dataclasses import dataclass         # 資料類別裝飾器
from enum import Enum                     # 列舉類型
from fastapi import FastAPI, Request, HTTPException  # Web框架
from fastapi.middleware.cors import CORSMiddleware   # 跨域處理
import uvicorn              # ASGI伺服器
from langchain_chroma import Chroma       # 向量資料庫
from langchain_huggingface import HuggingFaceEmbeddings  # 嵌入模型
from langchain_openai import ChatOpenAI   # OpenAI聊天模型
from langchain_community.utilities import GoogleSerperAPIWrapper  # 網路搜尋
from langchain.prompts import PromptTemplate  # 提示模板
from langchain.schema.runnable import RunnablePassthrough  # 可執行管道
from langchain.schema.output_parser import StrOutputParser  # 輸出解析器
from dotenv import load_dotenv            # 環境變數載入
import re                   # 正則表達式
import json                 # JSON處理
from functools import lru_cache          # 快取裝飾器
import time                 # 時間處理
import warnings             # 警告處理
```

### 2. 環境設定 (第27-35行)

```python
# 忽略 LangChain 棄用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 關閉 LangSmith 追蹤
os.environ["LANGCHAIN_TRACING_V2"] = "false"    # 關閉追蹤
os.environ["LANGCHAIN_ENDPOINT"] = ""           # 清空端點
os.environ["LANGCHAIN_API_KEY"] = ""            # 清空API金鑰
os.environ["LANGCHAIN_PROJECT"] = ""            # 清空專案名稱
```

### 3. 日誌設定 (第37-47行)

```python
# 設定詳細 log 檔案
log_filename = f"react_medical_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,                    # 日誌級別設為INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日誌格式
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # 檔案處理器
        logging.StreamHandler()            # 控制台處理器
    ]
)
```

### 4. 環境變數載入 (第49行)

```python
load_dotenv()  # 從.env檔案載入環境變數
```

### 5. 動作類型列舉 (第51-57行)

```python
class ActionType(Enum):
    """動作類型枚舉"""
    SEARCH_RAG = "search_rag"              # 搜尋本地RAG資料庫
    SEARCH_WEB = "search_web"              # 搜尋網路
    ANALYZE_QUERY = "analyze_query"        # 分析查詢
    FORMAT_RESPONSE = "format_response"    # 格式化回應
    VALIDATE_INFORMATION = "validate_information"  # 驗證資訊
    FINAL_ANSWER = "final_answer"          # 最終答案
```

### 6. ReAct步驟資料類別 (第59-67行)

```python
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

### 7. 查詢上下文資料類別 (第69-77行)

```python
@dataclass
class QueryContext:
    """查詢上下文"""
    query: str             # 原始查詢
    query_type: str        # 查詢類型：'doctor_info', 'medical_knowledge', 'general'
    priority_tool: str     # 優先工具：'rag', 'web_search', 'both'
    confidence_threshold: float = 0.7      # 信心度閾值，預設0.7
    requires_latest_info: bool = False     # 是否需要最新資訊，預設False
    medical_urgency: str = "low"           # 醫療急迫性：low, medium, high
```

### 8. 醫療查詢分析器類別 (第79-115行)

```python
class MedicalQueryAnalyzer:
    """醫療查詢分析器"""
    
    def __init__(self):
        # 定義各種關鍵字列表
        self.doctor_keywords = ['醫師', '醫生', '博士', '教授', '主任', '院長', '科主任']
        self.medical_keywords = ['治療', '症狀', '診斷', '藥物', '手術', '預防', '病因', '副作用']
        self.latest_keywords = ['最新', '2024', '2025', '近期', '現在', '目前', '新']
        self.urgent_keywords = ['急診', '緊急', '立即', '馬上', '危險', '嚴重']
    
    def analyze_query(self, query: str) -> QueryContext:
        """分析查詢意圖"""
        query_lower = query.lower()  # 轉小寫以便比對
        
        # 判斷醫療急迫性
        urgency = "high" if any(keyword in query for keyword in self.urgent_keywords) else "low"
        
        # 判斷是否需要最新資訊
        requires_latest = any(keyword in query for keyword in self.latest_keywords)
        
        # 根據關鍵字判斷查詢類型和優先工具
        if any(keyword in query for keyword in self.doctor_keywords):
            if requires_latest:
                return QueryContext(query, 'doctor_info', 'both', 
                                  requires_latest_info=True, medical_urgency=urgency)
            return QueryContext(query, 'doctor_info', 'rag', 
                              requires_latest_info=False, medical_urgency=urgency)
        
        if requires_latest:
            return QueryContext(query, 'medical_knowledge', 'web_search', 
                              requires_latest_info=True, medical_urgency=urgency)
        
        if any(keyword in query for keyword in self.medical_keywords):
            return QueryContext(query, 'medical_knowledge', 'both', 
                              requires_latest_info=False, medical_urgency=urgency)
        
        return QueryContext(query, 'general', 'both', 
                          requires_latest_info=False, medical_urgency=urgency)
```

### 9. ReAct醫療代理初始化 (第117-137行)

```python
class ReActMedicalAgent:
    """ReAct 醫療代理"""
    
    def __init__(self):
        self.analyzer = MedicalQueryAnalyzer()  # 創建查詢分析器
        self.cache = {}                        # 快取字典
        self.cache_ttl = timedelta(hours=1)    # 快取存活時間1小時
        self.react_steps = []                  # ReAct步驟列表
        self.max_iterations = 10               # 最大迭代次數
        
        # 初始化 LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 初始化工具
        self._init_tools()
        
        # 創建 ReAct Prompt
        self._create_react_prompt()
        
        logging.info("✅ ReAct 醫療代理初始化完成")
```

### 10. 工具初始化 (第138-160行)

```python
def _init_tools(self):
    """初始化工具"""
    try:
        # 初始化 RAG 系統
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",        # 使用BGE-M3嵌入模型
            model_kwargs={"device": "cpu"}    # 使用CPU設備
        )
        self.vectorstore = Chroma(
            persist_directory="chroma_db",    # 持久化目錄
            collection_name="doctors_smart_20250704_0608",  # 集合名稱
            embedding_function=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 初始化網路搜尋
        self.serper_wrapper = GoogleSerperAPIWrapper()
        
        logging.info("✅ 所有工具初始化成功")
    except Exception as e:
        logging.error(f"❌ 工具初始化失敗: {e}")
        raise
```

### 11. 創建ReAct提示模板 (第161-197行)

```python
def _create_react_prompt(self):
    """創建 ReAct Prompt"""
    self.react_prompt = PromptTemplate(
        input_variables=["query", "context", "previous_steps"],  # 輸入變數
        template="""
你是一位專業的醫療資訊助手，使用 ReAct (Reasoning and Acting) 方法來回答醫療問題。

用戶問題：{query}
查詢上下文：{context}
先前步驟：{previous_steps}

可用動作：
1. search_rag - 搜尋本地醫師資料庫
2. search_web - 搜尋網路最新資訊
3. analyze_query - 分析查詢意圖
4. format_response - 格式化回應
5. validate_information - 驗證資訊準確性
6. final_answer - 提供最終答案

請按照以下格式進行推理：

Thought: [你的思考過程，分析當前情況和下一步該做什麼]

Action: [選擇的動作]

Action Input: [動作的輸入參數]

Observation: [動作執行後的觀察結果]

Reasoning: [基於觀察結果的推理和分析]

如果你有足夠的資訊來回答問題，請使用 final_answer 動作。

現在開始：
"""
    )
```

### 12. 思考過程方法 (第198-233行)

```python
def _thought_process(self, query: str, context: QueryContext, 
                    previous_steps: List[ReActStep]) -> str:
    """思考過程"""
    if not previous_steps:
        return f"這是一個關於{context.query_type}的查詢，急迫性為{context.medical_urgency}。我需要先分析查詢的具體需求。"
    
    last_step = previous_steps[-1]  # 獲取最後一個步驟
    
    # 根據不同動作類型返回相應的思考內容
    if last_step.action == ActionType.ANALYZE_QUERY:
        if context.query_type == 'doctor_info':
            return "查詢已分析完成，這是醫師資訊查詢。我需要搜尋本地醫師資料庫。"
        elif context.requires_latest_info:
            return "查詢需要最新資訊，我應該先搜尋網路。"
        else:
            return "這是醫療知識查詢，我需要搜尋本地資料庫。"
    
    elif last_step.action == ActionType.SEARCH_RAG:
        if context.requires_latest_info or context.priority_tool == 'both':
            return "已獲得本地資料庫結果，現在需要搜尋網路以獲取最新資訊。"
        else:
            return "已獲得本地資料庫結果，現在需要驗證資訊準確性。"
    
    elif last_step.action == ActionType.SEARCH_WEB:
        if len([s for s in previous_steps if s.action == ActionType.SEARCH_RAG]) == 0:
            return "已獲得網路搜尋結果，現在需要搜尋本地資料庫以獲取更多資訊。"
        else:
            return "已獲得網路和本地資料，現在需要驗證資訊準確性。"
    
    elif last_step.action == ActionType.VALIDATE_INFORMATION:
        return "資訊驗證完成，現在需要格式化回應。"
    
    elif last_step.action == ActionType.FORMAT_RESPONSE:
        return "回應已格式化，準備提供最終答案。"
    
    return "需要分析當前狀況並決定下一步行動。"
```

### 13. 動作選擇方法 (第234-264行)

```python
def _select_action(self, query: str, context: QueryContext, 
                  previous_steps: List[ReActStep]) -> Tuple[ActionType, str]:
    """選擇動作"""
    if not previous_steps:
        return ActionType.ANALYZE_QUERY, query  # 第一步總是分析查詢
    
    completed_actions = [step.action for step in previous_steps]  # 已完成的動作
    
    # 按優先順序選擇下一個動作
    if ActionType.ANALYZE_QUERY not in completed_actions:
        return ActionType.ANALYZE_QUERY, query
    
    if context.query_type == 'doctor_info' and ActionType.SEARCH_RAG not in completed_actions:
        return ActionType.SEARCH_RAG, query
    
    if context.requires_latest_info and ActionType.SEARCH_WEB not in completed_actions:
        return ActionType.SEARCH_WEB, query
    
    if context.priority_tool == 'both':
        if ActionType.SEARCH_RAG not in completed_actions:
            return ActionType.SEARCH_RAG, query
        elif ActionType.SEARCH_WEB not in completed_actions:
            return ActionType.SEARCH_WEB, query
    
    if ActionType.VALIDATE_INFORMATION not in completed_actions:
        return ActionType.VALIDATE_INFORMATION, "驗證收集到的資訊"
    
    if ActionType.FORMAT_RESPONSE not in completed_actions:
        return ActionType.FORMAT_RESPONSE, "格式化最終回應"
    
    return ActionType.FINAL_ANSWER, "提供最終答案"
```

### 14. 動作執行方法 (第265-292行)

```python
async def _execute_action(self, action: ActionType, action_input: str, 
                        context: QueryContext) -> str:
    """執行動作"""
    try:
        # 根據動作類型調用相應的處理方法
        if action == ActionType.ANALYZE_QUERY:
            return await self._analyze_query_action(action_input)
        
        elif action == ActionType.SEARCH_RAG:
            return await self._search_rag_action(action_input)
        
        elif action == ActionType.SEARCH_WEB:
            return await self._search_web_action(action_input)
        
        elif action == ActionType.VALIDATE_INFORMATION:
            return await self._validate_information_action(action_input)
        
        elif action == ActionType.FORMAT_RESPONSE:
            return await self._format_response_action(action_input, context)
        
        elif action == ActionType.FINAL_ANSWER:
            return await self._final_answer_action(action_input, context)
        
        else:
            return f"未知動作: {action}"
            
    except Exception as e:
        return f"動作執行失敗: {str(e)}"
```

### 15. 查詢分析動作 (第293-310行)

```python
async def _analyze_query_action(self, query: str) -> str:
    """分析查詢動作"""
    # 使用 LLM 進行深度分析
    analysis_prompt = f"""
    請分析以下醫療查詢的詳細資訊：
    
    查詢: {query}
    
    請提供：
    1. 查詢的主要目的
    2. 所需資訊類型
    3. 預期回答格式
    4. 重要關鍵字
    """
    
    response = await asyncio.to_thread(self.llm.invoke, analysis_prompt)
    return f"查詢分析完成：{response.content}"
```

### 16. RAG搜尋動作 (第311-342行)

```python
async def _search_rag_action(self, query: str) -> str:
    """搜尋 RAG 動作"""
    try:
        # 搜尋相關文檔
        docs = await asyncio.to_thread(self.retriever.get_relevant_documents, query)
        
        if not docs:
            return "本地資料庫中未找到相關資訊"
        
        # 提取最相關的資訊
        relevant_info = []
        for doc in docs[:5]:  # 限制前5個最相關的文檔
            relevant_info.append(doc.page_content)
        
        combined_info = "\n".join(relevant_info)
        
        # 使用 LLM 處理和總結資訊
        summary_prompt = f"""
        請總結以下醫療資訊，回答用戶查詢：{query}
        
        相關資訊：
        {combined_info}
        
        請提供簡潔準確的總結：
        """
        
        response = await asyncio.to_thread(self.llm.invoke, summary_prompt)
        return f"本地資料庫搜尋結果：{response.content}"
        
    except Exception as e:
        return f"RAG 搜尋失敗：{str(e)}"
```

### 17. 網路搜尋動作 (第343-370行)

```python
async def _search_web_action(self, query: str) -> str:
    """搜尋網路動作"""
    try:
        # 優化搜尋查詢
        search_query = query
        if '醫師' in query or '醫生' in query:
            search_query += " 高雄醫學大學 醫師"  # 添加醫院關鍵字
        elif any(keyword in query for keyword in ['治療', '症狀', '診斷']):
            search_query += " 醫療 專業 2024"     # 添加專業關鍵字
        
        result = await asyncio.to_thread(self.serper_wrapper.run, search_query)
        
        # 使用 LLM 處理搜尋結果
        process_prompt = f"""
        請處理以下網路搜尋結果，回答用戶查詢：{query}
        
        搜尋結果：
        {result}
        
        請提供準確的資訊總結：
        """
        
        response = await asyncio.to_thread(self.llm.invoke, process_prompt)
        return f"網路搜尋結果：{response.content}"
        
    except Exception as e:
        return f"網路搜尋失敗：{str(e)}"
```

### 18. 資訊驗證動作 (第371-395行)

```python
async def _validate_information_action(self, action_input: str) -> str:
    """驗證資訊動作"""
    # 檢查之前步驟的結果
    rag_results = [step.observation for step in self.react_steps 
                  if step.action == ActionType.SEARCH_RAG]
    web_results = [step.observation for step in self.react_steps 
                  if step.action == ActionType.SEARCH_WEB]
    
    validation_prompt = f"""
    請驗證以下醫療資訊的一致性和準確性：
    
    本地資料結果：{rag_results}
    
    網路資料結果：{web_results}
    
    請檢查：
    1. 資訊是否一致
    2. 是否有衝突
    3. 資訊的可靠性
    4. 是否需要額外澄清
    """
    
    response = await asyncio.to_thread(self.llm.invoke, validation_prompt)
    return f"資訊驗證結果：{response.content}"
```

### 19. 格式化回應動作 (第396-435行)

```python
async def _format_response_action(self, action_input: str, context: QueryContext) -> str:
    """格式化回應動作 - 改進版"""
    # 收集所有相關資訊
    all_info = []
    rag_info = []
    web_info = []
    
    for step in self.react_steps:
        if step.action == ActionType.SEARCH_RAG:
            rag_info.append(step.observation)
        elif step.action == ActionType.SEARCH_WEB:
            web_info.append(step.observation)
        elif step.action == ActionType.VALIDATE_INFORMATION:
            all_info.append(step.observation)
    
    # 新的格式化 prompt - 更自然的回應
    format_prompt = f"""
    請將以下醫療資訊整理成自然、易讀的回應格式：
    
    用戶查詢：{context.query}
    
    本地資料：{' '.join(rag_info)}
    網路資料：{' '.join(web_info)}
    驗證結果：{' '.join(all_info)}
    
    請按照以下要求格式化回應：
    1. 使用自然流暢的中文
    2. 直接回答用戶問題，不要使用過多的標題和分項
    3. 保持專業但親切的語調
    4. 如果是醫師資訊，重點介紹專長和背景
    5. 如果是醫療知識，簡潔明了地解釋
    6. 避免過度結構化的格式
    7. 保持回答簡潔有力
    
    請直接提供整理後的回應內容，不要包含"格式化回應："等前綴。
    """
    
    response = await asyncio.to_thread(self.llm.invoke, format_prompt)
    return response.content
```

### 20. 最終答案動作 (第436-459行)

```python
async def _final_answer_action(self, action_input: str, context: QueryContext) -> str:
    """最終答案動作"""
    # 整合所有步驟的結果
    formatted_responses = [step.observation for step in self.react_steps 
                         if step.action == ActionType.FORMAT_RESPONSE]
    
    if formatted_responses:
        final_answer = formatted_responses[-1]  # 使用最後一個格式化回應
    else:
        # 如果沒有格式化回應，使用所有可用資訊
        all_observations = []
        for step in self.react_steps:
            if step.action in [ActionType.SEARCH_RAG, ActionType.SEARCH_WEB]:
                # 清理觀察結果，移除前綴
                clean_obs = step.observation.replace("本地資料庫搜尋結果：", "").replace("網路搜尋結果：", "")
                all_observations.append(clean_obs)
        
        final_answer = "\n\n".join(all_observations)
    
    # 添加醫療免責聲明
    disclaimer = "\n\n※ 以上資訊僅供參考"
    
    return final_answer + disclaimer
```

### 21. 創建推理方法 (第460-477行)

```python
def _create_reasoning(self, observation: str, action: ActionType) -> str:
    """創建推理"""
    reasoning_templates = {
        ActionType.ANALYZE_QUERY: "根據查詢分析結果，我了解了用戶的具體需求和查詢意圖。",
        ActionType.SEARCH_RAG: "本地資料庫搜尋提供了相關的醫療資訊，這些資訊來自可靠的醫療資料庫。",
        ActionType.SEARCH_WEB: "網路搜尋確保回答的時效性。",
        ActionType.VALIDATE_INFORMATION: "資訊驗證確保了回答的準確性和一致性。",
        ActionType.FORMAT_RESPONSE: "專業格式化使回答更加清晰和易於理解。",
        ActionType.FINAL_ANSWER: "整合所有資訊後提供了完整準確的最終答案。"
    }
    
    base_reasoning = reasoning_templates.get(action, "執行了相應的動作並獲得了結果。")
    
    if "失敗" in observation or "錯誤" in observation:
        return base_reasoning + " 但遇到了一些問題，需要調整策略。"
    
    return base_reasoning + " 結果符合預期，可以繼續下一步。"
```

### 22. 主要查詢方法 (第478-592行)

```python
async def query(self, question: str) -> Dict[str, Any]:
    """主要查詢方法 - ReAct 流程"""
    start_time = time.time()  # 記錄開始時間
    self.react_steps = []     # 清空步驟列表
    
    try:
        # 分析查詢上下文
        context = self.analyzer.analyze_query(question)
        logging.info(f"📝 查詢分析 - 類型: {context.query_type}, 優先工具: {context.priority_tool}")
        
        # ReAct 循環
        for step_num in range(1, self.max_iterations + 1):
            # Thought - 思考過程
            thought = self._thought_process(question, context, self.react_steps)
            logging.info(f"🤔 Step {step_num} - Thought: {thought}")
            
            # Action Selection - 動作選擇
            action, action_input = self._select_action(question, context, self.react_steps)
            logging.info(f"🎯 Step {step_num} - Action: {action.value}, Input: {action_input}")
            
            # Action Execution - 動作執行
            observation = await self._execute_action(action, action_input, context)
            logging.info(f"👀 Step {step_num} - Observation: {observation[:100]}...")
            
            # Reasoning - 推理過程
            reasoning = self._create_reasoning(observation, action)
            logging.info(f"💭 Step {step_num} - Reasoning: {reasoning}")
            
            # 記錄步驟
            react_step = ReActStep(
                step_number=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            self.react_steps.append(react_step)
            
            # 檢查是否完成
            if action == ActionType.FINAL_ANSWER:
                break
        
        # 提取最終答案
        final_step = self.react_steps[-1]
        final_answer = final_step.observation
        
        end_time = time.time()
        
        # 返回完整結果
        return {
            "result": final_answer,
            "query_type": context.query_type,
            "medical_urgency": context.medical_urgency,
            "react_steps": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action.value,
                    "action_input": step.action_input,
                    "observation": step.observation,
                    "reasoning": step.reasoning,
                    "timestamp": step.timestamp.isoformat()
                }
                for step in self.react_steps
            ],
            "total_steps": len(self.react_steps),
            "processing_time": round(end_time - start_time, 2),
            "success": True
        }
        
    except Exception as e:
        error_msg = f"ReAct 查詢處理錯誤: {str(e)}"
        logging.error(f"❌ {error_msg}")
        return {
            "result": "抱歉，查詢過程中發生錯誤，請稍後重試。",
            "error": error_msg,
            "react_steps": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action.value,
                    "action_input": step.action_input,
                    "observation": step.observation,
                    "reasoning": step.reasoning,
                    "timestamp": step.timestamp.isoformat()
                }
                for step in self.react_steps
            ],
            "success": False
        }
```

### 23. 全局代理實例創建 (第594-602行)

```python
# 創建全局 ReAct agent 實例
try:
    react_medical_agent = ReActMedicalAgent()
    logging.info("🏥 ReAct 醫療代理初始化成功")
except Exception as e:
    logging.error(f"❌ ReAct 醫療代理初始化失敗: {e}")
    react_medical_agent = None
```

### 24. FastAPI應用設定 (第604-618行)

```python
# FastAPI 設定
app = FastAPI(
    title="ReAct 醫療資訊查詢系統",
    description="使用 ReAct 框架的專業醫療資訊查詢 API",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 允許所有來源
    allow_credentials=True,   # 允許憑證
    allow_methods=["*"],      # 允許所有方法
    allow_headers=["*"],      # 允許所有標頭
)
```

### 25. 根路徑端點 (第620-628行)

```python
@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "ReAct 醫療資訊查詢系統 API",
        "version": "3.0",
        "framework": "ReAct (Reasoning and Acting)",
        "status": "運行中" if react_medical_agent else "初始化失敗"
    }
```

### 26. 主要查詢API端點 (第630-648行)

```python
@app.post("/api/ask")
async def ask_react_medical_agent(request: Request):
    """ReAct 醫療資訊查詢 API"""
    if not react_medical_agent:
        raise HTTPException(status_code=503, detail="ReAct 醫療查詢系統未能正確初始化")
    
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="問題不能為空")
        
        logging.info(f"🚀 收到 ReAct 查詢: {question}")
        
        # 使用 ReAct 方法處理查詢
        result = await react_medical_agent.query(question)
        
        logging.info(f"✅ ReAct 查詢完成，總步驟: {result.get('total_steps', 0)}, 耗時: {result.get('processing_time', 0)}秒")
        
        return result
        
    except Exception as e:
        error_msg = f"ReAct API 錯誤: {str(e)}"
        logging.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
```

### 27. 健康檢查端點 (第650-663行)

```python
@app.get("/api/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy" if react_medical_agent else "unhealthy",
        "framework": "ReAct (Reasoning and Acting)",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(react_medical_agent.cache) if react_medical_agent else 0,
        "system_info": {
            "rag_system": "運行中" if react_medical_agent and react_medical_agent.vectorstore else "未初始化",
            "web_search": "運行中" if react_medical_agent and react_medical_agent.serper_wrapper else "未初始化",
            "max_iterations": react_medical_agent.max_iterations if react_medical_agent else 0
        }
    }
```

### 28. 追蹤記錄端點 (第665-669行)

```python
@app.get("/api/react-trace/{trace_id}")
async def get_react_trace(trace_id: str):
    """獲取 ReAct 追蹤記錄"""
    # 這裡可以實現追蹤記錄的存儲和檢索
    return {"message": "ReAct 追蹤記錄功能待實現"}
```

### 29. 清除快取端點 (第671-679行)

```python
@app.delete("/api/cache")
async def clear_cache():
    """清除快取"""
    if not react_medical_agent:
        raise HTTPException(status_code=503, detail="ReAct 醫療查詢系統未能正確初始化")
    
    cache_size = len(react_medical_agent.cache)
    react_medical_agent.cache.clear()
    return {"message": f"已清除 {cache_size} 個快取項目"}
```

### 30. 伺服器運行函數 (第681-695行)

```python
def run_server():
    """運行服務器"""
    logging.info("🏥 啟動 ReAct 醫療資訊查詢系統")
    logging.info("🧠 框架: ReAct (Reasoning and Acting)")
    logging.info("📍 服務地址: http://localhost:8000")
    logging.info("🔗 API 端點: http://localhost:8000/api/ask")
    logging.info("💊 健康檢查: http://localhost:8000/api/health")
    logging.info(f"📝 Log 檔案: {log_filename}")
    logging.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",    # 監聽所有網路介面
        port=8000,         # 端口8000
        log_level="info"   # 日誌級別
    )

if __name__ == "__main__":
    run_server()  # 直接執行時啟動伺服器
```

## 🚨 急迫性分析機制

### 為什麼要設置急迫性？

分析醫療查詢的急迫性具有非常重要的意義，主要體現在以下幾個方面：

#### 1. 🔍 急迫性識別機制

```python
# 第86行：定義急迫性關鍵字
self.urgent_keywords = ['急診', '緊急', '立即', '馬上', '危險', '嚴重']

# 第93-94行：判斷急迫性
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

#### 6. 🔄 與其他參數的協同作用

```python
@dataclass
class QueryContext:
    medical_urgency: str = "low"  # 與其他參數協同
    confidence_threshold: float = 0.7
    requires_latest_info: bool = False
```

**協同效應：**
- **高急迫性 + 需要最新資訊**：優先網路搜尋
- **高急迫性 + 低信心度**：提供緊急建議，同時建議專業醫療諮詢
- **低急迫性 + 高信心度**：提供詳細、經過驗證的資訊

#### 7. 🎯 未來擴展可能性

```python
# 可以擴展為更細緻的急迫性等級
medical_urgency: str = "low"  # 可擴展為 "low", "medium", "high", "critical"
```

**擴展方向：**
- **critical**：生命危險，需要立即急診
- **high**：需要快速醫療干預
- **medium**：需要醫療評估但不急迫
- **low**：一般諮詢或預防性問題

### 急迫性設置的意義總結

1. **🔴 醫療安全**：確保緊急情況得到優先處理
2. **⚡ 回應效率**：根據急迫性調整處理策略
3. **🎯 用戶體驗**：提供符合期望的回應速度和內容
4. **🏥 專業性**：體現醫療系統的專業判斷能力
5. **⚙️ 系統優化**：根據急迫性調整資源分配

這種設計讓系統能夠像真正的醫療專業人員一樣，根據情況的緊急程度做出相應的處理決策，確保用戶在緊急情況下能夠得到及時、準確的醫療建議。

## 🏗️ 系統架構

### 核心組件

1. **MedicalQueryAnalyzer**：查詢分析器
2. **ReActMedicalAgent**：主要代理類別
3. **ActionType**：動作類型列舉
4. **QueryContext**：查詢上下文
5. **ReActStep**：ReAct步驟記錄

### 工作流程

1. **查詢接收** → 2. **意圖分析** → 3. **ReAct循環** → 4. **結果返回**

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

**版本**：3.0  
**框架**：ReAct (Reasoning and Acting)  
**更新日期**：2024年
