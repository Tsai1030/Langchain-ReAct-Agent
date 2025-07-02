import json
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
import os
# 讀取 .env

from dotenv import load_dotenv
load_dotenv()
# 確認 API Key 有讀取到
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY")[:20] + "..." if os.getenv("OPENAI_API_KEY") else "未設定")

# 1. 查詢醫師專長的自定義函數
def search_doctor_by_specialty(specialty: str) -> str:
    with open('doctors.json', 'r', encoding='utf-8') as f:
        doctors = json.load(f)
    result = []
    for doc in doctors:
        if specialty in doc.get("specialty", []):
            result.append(f"{doc['name']}（{doc['department']}）")
    if not result:
        return f"查無專長為「{specialty}」的醫師。"
    return "、".join(result)

# 2. 包裝成 LangChain Tool
search_specialty_tool = Tool(
    name="查詢醫師專長",
    func=search_doctor_by_specialty,
    description="輸入一個專長名稱，查詢有哪些醫師有這個專長。例如：高血壓"
)

# 3. 建立 Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_specialty_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. 測試
if __name__ == "__main__":
    result = agent.run("請幫我查詢有哪些醫師專長是高血壓？")
    print(result) 