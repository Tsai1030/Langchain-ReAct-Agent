import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_chroma import Chroma  # 更新的套件
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import re
import json
from functools import lru_cache
import time
import warnings

# 忽略 LangChain 棄用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 關閉 LangSmith 追蹤
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# 設定詳細 log 檔案
log_filename = f"medical_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 讀取 .env
load_dotenv()

@dataclass
class QueryContext:
    """查詢上下文"""
    query: str
    query_type: str  # 'doctor_info', 'medical_knowledge', 'general'
    priority_tool: str  # 'rag', 'web_search', 'both'
    confidence_threshold: float = 0.7

class MedicalQueryAnalyzer:
    """醫療查詢分析器"""
    
    def __init__(self):
        self.doctor_keywords = ['醫師', '醫生', '博士', '教授', '主任', '院長', '科主任']
        self.medical_keywords = ['治療', '症狀', '診斷', '藥物', '手術', '預防', '病因']
        self.latest_keywords = ['最新', '2024', '2025', '近期', '現在', '目前', '新']
    
    def analyze_query(self, query: str) -> QueryContext:
        """分析查詢意圖"""
        query_lower = query.lower()
        
        # 判斷是否查詢醫師資訊
        if any(keyword in query for keyword in self.doctor_keywords):
            if any(keyword in query for keyword in self.latest_keywords):
                return QueryContext(query, 'doctor_info', 'both')
            return QueryContext(query, 'doctor_info', 'rag')
        
        # 判斷是否查詢最新醫療資訊
        if any(keyword in query for keyword in self.latest_keywords):
            return QueryContext(query, 'medical_knowledge', 'web_search')
        
        # 判斷是否查詢醫療知識
        if any(keyword in query for keyword in self.medical_keywords):
            return QueryContext(query, 'medical_knowledge', 'both')
        
        # 默認情況
        return QueryContext(query, 'general', 'both')

class MedicalResponseFormatter:
    """醫療回應格式化器"""
    
    @staticmethod
    def format_doctor_info(response: str) -> str:
        """格式化醫師資訊"""
        if not response:
            return "抱歉，未找到相關醫師資訊。"
        
        # 移除重複資訊
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        
        return '\n'.join(unique_lines)
    
    @staticmethod
    def format_medical_info(response: str) -> str:
        """格式化醫療資訊"""
        if not response:
            return "抱歉，未找到相關醫療資訊。"
        
        # 確保資訊的專業性和準確性
        if len(response) < 50:
            return response + "\n\n建議您諮詢專業醫師以獲得更詳細的建議。"
        
        return response + "\n\n※ 以上資訊僅供參考，實際診斷和治療請諮詢專業醫師。"

class ImprovedMedicalAgent:
    """改進版醫療查詢代理"""
    
    def __init__(self):
        self.analyzer = MedicalQueryAnalyzer()
        self.formatter = MedicalResponseFormatter()
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # 初始化 LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 初始化 RAG 系統
        self._init_rag_system()
        
        # 初始化 Web Search
        self._init_web_search()
        
        # 創建自訂義 prompt
        self._create_custom_prompts()
        
        # 創建查詢鏈
        self._create_query_chains()
        
        logging.info("✅ 醫療查詢系統初始化完成")
    
    def _init_rag_system(self):
        """初始化 RAG 系統"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"}
            )
            self.vectorstore = Chroma(
                persist_directory="chroma_db",
                collection_name="doctors_smart_20250704_0608",
                embedding_function=self.embedding_model
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            logging.info("✅ RAG 系統初始化成功")
        except Exception as e:
            logging.error(f"❌ RAG 系統初始化失敗: {e}")
            raise
    
    def _init_web_search(self):
        """初始化網路搜尋"""
        try:
            self.serper_wrapper = GoogleSerperAPIWrapper()
            logging.info("✅ 網路搜尋初始化成功")
        except Exception as e:
            logging.error(f"❌ 網路搜尋初始化失敗: {e}")
            # 不拋出錯誤，允許系統在沒有網路搜尋的情況下運行
            self.serper_wrapper = None
    
    def _create_custom_prompts(self):
        """創建自訂義 prompt"""
        
        # 醫師資訊查詢 prompt
        self.doctor_info_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
你是一位專業的醫療資訊助手。請基於以下資料庫內容，回答用戶關於醫師的問題。

用戶問題：{query}

相關資料：
{context}

請按以下格式回答：
【醫師資訊】
• 醫師姓名：
• 專長領域：
• 學歷背景：
• 職務經歷：
• 聯絡資訊：（如有）

回答要求：
- 資訊要準確完整
- 如果資料不足，請明確說明
- 使用繁體中文回答
- 保持專業醫療用語
"""
        )
        
        # 醫療知識查詢 prompt
        self.medical_knowledge_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
你是一位專業的醫療資訊助手。請基於提供的資訊，回答用戶的醫療問題。

用戶問題：{query}

相關資訊：
{context}

請按以下格式回答：
【醫療資訊】
• 簡要說明：
• 詳細內容：
• 注意事項：
• 專業建議：

回答要求：
- 資訊要科學準確
- 避免給出具體診斷建議
- 強調諮詢專業醫師的重要性
- 使用繁體中文回答
- 如涉及藥物或治療，請特別標註風險
"""
        )
        
        # 整合回答 prompt
        self.integration_prompt = PromptTemplate(
            input_variables=["query", "rag_result", "web_result"],
            template="""
作為專業醫療資訊助手，請整合以下資料來回答用戶問題：

用戶問題：{query}

本地資料庫結果：
{rag_result}

網路搜尋結果：
{web_result}

請提供一個完整、準確的回答，要求：
1. 整合兩個來源的資訊
2. 突出重點和關鍵資訊
3. 如有衝突，請說明
4. 保持專業性和準確性
5. 使用繁體中文回答

【整合回答】
（完整的回答內容）

【資料來源】
• 本地醫師資料庫：{rag_result}
• 網路最新資訊：{web_result}

【重要提醒】
以上資訊僅供參考，實際診斷和治療請諮詢專業醫師。
"""
        )
    
    def _create_query_chains(self):
        """創建查詢鏈"""
        
        # RAG 查詢鏈
        self.rag_chain = (
            {
                "context": self.retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
                "query": RunnablePassthrough()
            }
            | self.doctor_info_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 醫療知識查詢鏈
        self.medical_chain = (
            {
                "context": self.retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
                "query": RunnablePassthrough()
            }
            | self.medical_knowledge_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _rag_query(self, query: str) -> str:
        """RAG 查詢"""
        logging.info(f"🔍 RAG 查詢: {query}")
        
        # 檢查快取
        cache_key = f"rag_{hash(query)}"
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                logging.info("📋 使用快取結果")
                return cached_result
        
        try:
            # 判斷查詢類型並使用對應的鏈
            if any(keyword in query for keyword in ['醫師', '醫生', '博士']):
                result = self.rag_chain.invoke(query)
            else:
                result = self.medical_chain.invoke(query)
            
            # 快取結果
            self.cache[cache_key] = (result, datetime.now())
            
            logging.info(f"✅ RAG 查詢完成")
            return result
            
        except Exception as e:
            error_msg = f"RAG 查詢錯誤: {str(e)}"
            logging.error(f"❌ {error_msg}")
            return f"抱歉，本地資料庫查詢失敗：{error_msg}"
    
    def _web_search(self, query: str) -> str:
        """網路搜尋"""
        logging.info(f"🌐 網路搜尋: {query}")
        
        if not self.serper_wrapper:
            return "網路搜尋功能暫時不可用"
        
        # 檢查快取
        cache_key = f"web_{hash(query)}"
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                logging.info("📋 使用快取結果")
                return cached_result
        
        try:
            # 優化搜尋查詢
            search_query = query
            if '醫師' in query or '醫生' in query:
                search_query += " 高雄醫學大學 醫師"
            elif any(keyword in query for keyword in ['治療', '症狀', '診斷']):
                search_query += " 醫療 專業"
            
            result = self.serper_wrapper.run(search_query)
            
            # 快取結果
            self.cache[cache_key] = (result, datetime.now())
            
            logging.info(f"✅ 網路搜尋完成")
            return result
            
        except Exception as e:
            error_msg = f"網路搜尋錯誤: {str(e)}"
            logging.error(f"❌ {error_msg}")
            return f"抱歉，網路搜尋失敗：{error_msg}"
    
    async def query(self, question: str) -> Dict[str, Any]:
        """主要查詢方法"""
        start_time = time.time()
        
        # 分析查詢
        context = self.analyzer.analyze_query(question)
        logging.info(f"📝 查詢分析 - 類型: {context.query_type}, 優先工具: {context.priority_tool}")
        
        try:
            # 智能查詢策略
            if context.priority_tool == 'rag':
                result = await self._rag_only_query(question)
            elif context.priority_tool == 'web_search':
                result = await self._web_only_query(question)
            else:
                result = await self._integrated_query(question)
            
            # 格式化回答
            if context.query_type == 'doctor_info':
                formatted_result = self.formatter.format_doctor_info(result)
            else:
                formatted_result = self.formatter.format_medical_info(result)
            
            end_time = time.time()
            
            return {
                "result": formatted_result,
                "query_type": context.query_type,
                "priority_tool": context.priority_tool,
                "processing_time": round(end_time - start_time, 2),
                "success": True
            }
            
        except Exception as e:
            error_msg = f"查詢處理錯誤: {str(e)}"
            logging.error(f"❌ {error_msg}")
            return {
                "result": "抱歉，查詢過程中發生錯誤，請稍後重試。",
                "error": error_msg,
                "success": False
            }
    
    async def _rag_only_query(self, question: str) -> str:
        """僅使用 RAG 查詢"""
        return await asyncio.to_thread(self._rag_query, question)
    
    async def _web_only_query(self, question: str) -> str:
        """僅使用網路搜尋"""
        return await asyncio.to_thread(self._web_search, question)
    
    async def _integrated_query(self, question: str) -> str:
        """整合查詢"""
        # 並行執行兩個查詢
        rag_task = asyncio.create_task(self._rag_only_query(question))
        web_task = asyncio.create_task(self._web_only_query(question))
        
        try:
            rag_result, web_result = await asyncio.gather(rag_task, web_task)
            
            # 使用整合 prompt
            integration_prompt = self.integration_prompt.format(
                query=question,
                rag_result=rag_result,
                web_result=web_result
            )
            
            response = await asyncio.to_thread(self.llm.invoke, integration_prompt)
            return response.content
            
        except Exception as e:
            logging.error(f"整合查詢錯誤: {e}")
            # 降級策略：如果整合失敗，至少回傳 RAG 結果
            try:
                return await rag_task
            except:
                return "抱歉，查詢過程中發生錯誤。"

# 創建全局 agent 實例
try:
    medical_agent = ImprovedMedicalAgent()
    logging.info("🏥 醫療查詢代理初始化成功")
except Exception as e:
    logging.error(f"❌ 醫療查詢代理初始化失敗: {e}")
    medical_agent = None

# FastAPI 設定
app = FastAPI(
    title="醫療資訊查詢系統",
    description="專業醫療資訊查詢 API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "醫療資訊查詢系統 API",
        "version": "2.0",
        "status": "運行中" if medical_agent else "初始化失敗"
    }

@app.post("/api/ask")
async def ask_medical_agent(request: Request):
    """醫療資訊查詢 API"""
    if not medical_agent:
        raise HTTPException(status_code=503, detail="醫療查詢系統未能正確初始化")
    
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="問題不能為空")
        
        logging.info(f"🚀 收到查詢: {question}")
        
        # 使用改進的查詢方法
        result = await medical_agent.query(question)
        
        logging.info(f"✅ 查詢完成，耗時: {result.get('processing_time', 0)}秒")
        
        return result
        
    except Exception as e:
        error_msg = f"API 錯誤: {str(e)}"
        logging.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy" if medical_agent else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(medical_agent.cache) if medical_agent else 0,
        "system_info": {
            "rag_system": "運行中" if medical_agent and medical_agent.vectorstore else "未初始化",
            "web_search": "運行中" if medical_agent and medical_agent.serper_wrapper else "未初始化"
        }
    }

@app.delete("/api/cache")
async def clear_cache():
    """清除快取"""
    if not medical_agent:
        raise HTTPException(status_code=503, detail="醫療查詢系統未能正確初始化")
    
    cache_size = len(medical_agent.cache)
    medical_agent.cache.clear()
    return {"message": f"已清除 {cache_size} 個快取項目"}

# 用於直接執行的函數
def run_server():
    """運行服務器"""
    logging.info("🏥 啟動改進版醫療資訊查詢系統")
    logging.info("📍 服務地址: http://localhost:8000")
    logging.info("🔗 API 端點: http://localhost:8000/api/ask")
    logging.info("💊 健康檢查: http://localhost:8000/api/health")
    logging.info(f"📝 Log 檔案: {log_filename}")
    logging.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()