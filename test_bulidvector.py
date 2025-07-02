from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# 讀取 .env
load_dotenv()
# 確認 API Key 有讀取到
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY")[:20] + "..." if os.getenv("OPENAI_API_KEY") else "未設定")

# 1. 設定 bge-m3 embedding


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory="chroma_db",        # 應該是 "chroma_db"，不是 "chroma_db/doctorv4"
    collection_name="doctorv4",           # 應該是 "doctorv4"
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.invoke({"query": "請問這個資料庫裡有什麼醫生？"})
print(result)

