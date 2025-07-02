import os
import sys
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# 關閉 LangSmith 追蹤
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# 設定詳細 log 檔案
log_filename = f"agent_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同時顯示在終端機
    ]
)

# 讀取 .env
load_dotenv()

# 1. RAG 設定
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="doctorv4",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

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

rag_tool = Tool(
    name="醫師資料庫查詢",
    func=rag_query,
    description="查詢本地醫師資料庫，包含醫師姓名、專長、學歷、經歷、職稱等資訊。輸入格式：醫師姓名 + 查詢項目，例如：'朱志生醫師專長'、'林宗翰醫師學歷'、'高醫心臟科醫師'。這個工具會回傳醫師的詳細資訊。"
)

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

if __name__ == "__main__":
    logging.info(" 啟動醫療資訊查詢系統 FastAPI 後端")
    logging.info("📍 服務地址: http://localhost:8000")
    logging.info("🔗 API 端點: http://localhost:8000/api/ask")
    logging.info(f"📝 詳細 log 檔案: {log_filename}")
    logging.info("=" * 60)
    uvicorn.run("react_agent_api:app", host="0.0.0.0", port=8000, reload=True) 