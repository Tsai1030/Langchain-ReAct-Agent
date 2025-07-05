import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
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
log_filename = f"react_medical_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

class ActionType(Enum):
    """動作類型枚舉"""
    SEARCH_RAG = "search_rag"
    SEARCH_WEB = "search_web"
    ANALYZE_QUERY = "analyze_query"
    FORMAT_RESPONSE = "format_response"
    VALIDATE_INFORMATION = "validate_information"
    FINAL_ANSWER = "final_answer"

@dataclass
class ReActStep:
    """ReAct 步驟"""
    step_number: int
    thought: str
    action: ActionType
    action_input: str
    observation: str
    reasoning: str
    timestamp: datetime

@dataclass
class QueryContext:
    """查詢上下文"""
    query: str
    query_type: str  # 'doctor_info', 'medical_knowledge', 'general'
    priority_tool: str  # 'rag', 'web_search', 'both'
    confidence_threshold: float = 0.7
    requires_latest_info: bool = False
    medical_urgency: str = "low"  # low, medium, high

class MedicalQueryAnalyzer:
    """醫療查詢分析器"""
    
    def __init__(self):
        self.doctor_keywords = ['醫師', '醫生', '博士', '教授', '主任', '院長', '科主任']
        self.medical_keywords = ['治療', '症狀', '診斷', '藥物', '手術', '預防', '病因', '副作用']
        self.latest_keywords = ['最新', '2024', '2025', '近期', '現在', '目前', '新']
        self.urgent_keywords = ['急診', '緊急', '立即', '馬上', '危險', '嚴重']
    
    def analyze_query(self, query: str) -> QueryContext:
        """分析查詢意圖"""
        query_lower = query.lower()
        
        # 判斷醫療急迫性
        urgency = "high" if any(keyword in query for keyword in self.urgent_keywords) else "low"
        
        # 判斷是否需要最新資訊
        requires_latest = any(keyword in query for keyword in self.latest_keywords)
        
        # 判斷查詢類型
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

class ReActMedicalAgent:
    """ReAct 醫療代理"""
    
    def __init__(self):
        self.analyzer = MedicalQueryAnalyzer()
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.react_steps = []
        self.max_iterations = 10
        
        # 初始化 LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 初始化工具
        self._init_tools()
        
        # 創建 ReAct Prompt
        self._create_react_prompt()
        
        logging.info("✅ ReAct 醫療代理初始化完成")
    
    def _init_tools(self):
        """初始化工具"""
        try:
            # 初始化 RAG 系統
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"}
            )
            self.vectorstore = Chroma(
                persist_directory="chroma_db",
                collection_name="doctors_smart_20250704_0608",
                embedding_function=self.embedding_model
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            
            # 初始化網路搜尋
            self.serper_wrapper = GoogleSerperAPIWrapper()
            
            logging.info("✅ 所有工具初始化成功")
        except Exception as e:
            logging.error(f"❌ 工具初始化失敗: {e}")
            raise
    
    def _create_react_prompt(self):
        """創建 ReAct Prompt"""
        self.react_prompt = PromptTemplate(
            input_variables=["query", "context", "previous_steps"],
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
    
    def _thought_process(self, query: str, context: QueryContext, 
                        previous_steps: List[ReActStep]) -> str:
        """思考過程"""
        if not previous_steps:
            return f"這是一個關於{context.query_type}的查詢，急迫性為{context.medical_urgency}。我需要先分析查詢的具體需求。"
        
        last_step = previous_steps[-1]
        
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
    
    def _select_action(self, query: str, context: QueryContext, 
                      previous_steps: List[ReActStep]) -> Tuple[ActionType, str]:
        """選擇動作"""
        if not previous_steps:
            return ActionType.ANALYZE_QUERY, query
        
        completed_actions = [step.action for step in previous_steps]
        
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
    
    async def _execute_action(self, action: ActionType, action_input: str, 
                            context: QueryContext) -> str:
        """執行動作"""
        try:
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
    
    async def _search_web_action(self, query: str) -> str:
        """搜尋網路動作"""
        try:
            # 優化搜尋查詢
            search_query = query
            if '醫師' in query or '醫生' in query:
                search_query += " 高雄醫學大學 醫師"
            elif any(keyword in query for keyword in ['治療', '症狀', '診斷']):
                search_query += " 醫療 專業 2024"
            
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
    
    async def _final_answer_action(self, action_input: str, context: QueryContext) -> str:
        """最終答案動作"""
        # 整合所有步驟的結果
        formatted_responses = [step.observation for step in self.react_steps 
                             if step.action == ActionType.FORMAT_RESPONSE]
        
        if formatted_responses:
            final_answer = formatted_responses[-1]
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
    
    async def query(self, question: str) -> Dict[str, Any]:
        """主要查詢方法 - ReAct 流程"""
        start_time = time.time()
        self.react_steps = []
        
        try:
            # 分析查詢上下文
            context = self.analyzer.analyze_query(question)
            logging.info(f"📝 查詢分析 - 類型: {context.query_type}, 優先工具: {context.priority_tool}")
            
            # ReAct 循環
            for step_num in range(1, self.max_iterations + 1):
                # Thought
                thought = self._thought_process(question, context, self.react_steps)
                logging.info(f"🤔 Step {step_num} - Thought: {thought}")
                
                # Action Selection
                action, action_input = self._select_action(question, context, self.react_steps)
                logging.info(f"🎯 Step {step_num} - Action: {action.value}, Input: {action_input}")
                
                # Action Execution
                observation = await self._execute_action(action, action_input, context)
                logging.info(f"👀 Step {step_num} - Observation: {observation[:100]}...")
                
                # Reasoning
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

# 創建全局 ReAct agent 實例
try:
    react_medical_agent = ReActMedicalAgent()
    logging.info("🏥 ReAct 醫療代理初始化成功")
except Exception as e:
    logging.error(f"❌ ReAct 醫療代理初始化失敗: {e}")
    react_medical_agent = None

# FastAPI 設定
app = FastAPI(
    title="ReAct 醫療資訊查詢系統",
    description="使用 ReAct 框架的專業醫療資訊查詢 API",
    version="3.0"
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
        "message": "ReAct 醫療資訊查詢系統 API",
        "version": "3.0",
        "framework": "ReAct (Reasoning and Acting)",
        "status": "運行中" if react_medical_agent else "初始化失敗"
    }

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

@app.get("/api/react-trace/{trace_id}")
async def get_react_trace(trace_id: str):
    """獲取 ReAct 追蹤記錄"""
    # 這裡可以實現追蹤記錄的存儲和檢索
    return {"message": "ReAct 追蹤記錄功能待實現"}

@app.delete("/api/cache")
async def clear_cache():
    """清除快取"""
    if not react_medical_agent:
        raise HTTPException(status_code=503, detail="ReAct 醫療查詢系統未能正確初始化")
    
    cache_size = len(react_medical_agent.cache)
    react_medical_agent.cache.clear()
    return {"message": f"已清除 {cache_size} 個快取項目"}

# 用於直接執行的函數
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
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()