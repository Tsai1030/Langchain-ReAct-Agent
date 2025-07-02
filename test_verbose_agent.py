# test_verbose_agent.py
import os
import sys
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

# 強制設定詳細 log
os.environ["LANGCHAIN_VERBOSE"] = "true"

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
    print(f"🔍 RAG 檢索開始: {query}")
    print(f"📊 向量資料庫查詢: {query}")
    docs = vectorstore.similarity_search(query, k=3)
    print(f" 找到 {len(docs)} 個相關文件")
    for i, doc in enumerate(docs):
        print(f"   文件 {i+1}: {doc.page_content[:100]}...")
    
    result = qa_chain.invoke({"query": query})
    response = result['result'] if isinstance(result, dict) else str(result)
    print(f"🤖 RAG 最終結果: {response[:200]}...")
    return response

rag_tool = Tool(
    name="醫師資料庫查詢",
    func=rag_query,
    description="查詢本地醫師資料庫，包含醫師姓名、專長、學歷、經歷、職稱等資訊。"
)

# 2. Web Search Tool
serper_wrapper = GoogleSerperAPIWrapper()
def smart_web_search(query: str) -> str:
    print(f"🌐 Web Search 開始: {query}")
    try:
        result = serper_wrapper.run(query)
        print(f"🌐 Web Search 結果: {result[:200]}...")
        return result
    except Exception as e:
        error_msg = f"Web search failed: {e}"
        print(f"❌ Web Search 錯誤: {error_msg}")
        return error_msg

web_search_tool = Tool(
    name="網路搜尋",
    func=smart_web_search,
    description="查詢網路最新資訊，適合查詢醫療新知、最新治療方法、研究報告等。"
)

# 3. Agent
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

# 4. 測試
if __name__ == "__main__":
    print("🧪 醫療資訊查詢系統 - 詳細調試模式")
    print("=" * 60)
    print("🧪 提示：")
    print("  - 輸入 'quit' 或 'exit' 退出")
    print("  - 輸入 'help' 查看範例查詢")
    print("  - 輸入 'test' 執行預設測試")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n請輸入查詢問題: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再見！")
                break
            elif question.lower() == 'help':
                print("\n📋 範例查詢：")
                print("  - 朱志生醫師專長")
                print("  - 林宗翰醫師學歷")
                print("  - 2024年高血壓治療新趨勢")
                print("  - 心臟病預防方法")
                print("  - 高醫心臟科醫師")
                continue
            elif question.lower() == 'test':
                question = "朱志生醫師專長"
                print(f"🧪 執行預設測試: {question}")
            
            if not question:
                print("❌ 請輸入有效的查詢問題")
                continue
                
            print(f"\n🚀 開始處理查詢: {question}")
            print("=" * 60)
            
            result = agent.run(question)
            
            print("=" * 60)
            print(f"✅ 查詢完成: {result}")
            print("\n")
            
        except KeyboardInterrupt:
            print("\n👋 再見！")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            print("請檢查網路連線和 API 設定")