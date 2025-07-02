# RAG-Agent-Websearch

本專案是一個結合 Retrieval-Augmented Generation (RAG) 與 ReAct Agent 架構的醫療資訊查詢系統。  
後端採用 FastAPI，整合本地醫師資料庫（Chroma 向量庫）與網路搜尋（Google Serper API），  
能根據用戶問題自動選擇最合適的查詢方式，並以大型語言模型（LLM）生成專業回覆。  
前端可自訂（預設支援 React/Vite）。

---

## 目錄

- [專案特色](#專案特色)
- [ReAct Agent 架構與原理](#react-agent-架構與原理)
- [主要檔案說明](#主要檔案說明)
- [安裝與操作步驟](#安裝與操作步驟)
  - [1. Python 環境安裝](#1-python-環境安裝)
  - [2. 取得 API 金鑰](#2-取得-api-金鑰)
  - [3. 設定環境變數](#3-設定環境變數)
  - [4. 安裝依賴套件](#4-安裝依賴套件)
  - [5. 建立本地向量資料庫](#5-建立本地向量資料庫)
  - [6. 啟動 FastAPI 後端](#6-啟動-fastapi-後端)
  - [7. 前端啟動（可選）](#7-前端啟動可選)
  - [8. 測試](#8-測試)
- [API 說明](#api-說明)
- [前端串接範例](#前端串接範例)
- [技術棧與設計邏輯](#技術棧與設計邏輯)
- [常見問題排查](#常見問題排查)
- [聯絡方式](#聯絡方式)

---

## 專案特色

- **RAG + ReAct Agent**：結合本地知識檢索與網路即時搜尋，智能決策查詢路徑。
- **多工具整合**：可查詢醫師專長、學歷、經歷，也能搜尋醫療新知。
- **高可擴展性**：可輕鬆擴充資料庫或新增工具。
- **詳細日誌**：每次查詢過程完整記錄，方便除錯與追蹤。

---

## ReAct Agent 架構與原理

### 什麼是 ReAct Agent？

ReAct（Reason + Act）Agent 是一種結合「推理」與「行動」的智能體架構。  
它能根據用戶問題，**自動選擇合適的工具**（如本地資料庫查詢、網路搜尋），並將多個工具的結果整合，產生最終回覆。

### 本專案的 ReAct Agent 流程

1. **接收用戶問題**：如「林宗翰醫師的專長是什麼？」或「2024年高血壓治療新趨勢」。
2. **推理判斷**：Agent 會根據問題內容，決定要用哪個工具：
   - 若問題與本地醫師資料相關，優先查詢向量資料庫（RAG）。
   - 若問題需最新醫療資訊，則呼叫網路搜尋工具。
   - 若問題複雜，則可多次交互使用兩種工具。
3. **行動（Action）**：Agent 執行查詢，取得資料。
4. **整合回覆**：LLM 將查詢結果彙整，生成自然語言答案。

### 使用的 LangChain Tool

- **醫師資料庫查詢（RAG Tool）**  
  - 工具名稱：`醫師資料庫查詢`
  - 功能：查詢本地向量資料庫，檢索醫師姓名、專長、學歷、經歷等。
  - 實作：`rag_query` 函數，結合 HuggingFace Embeddings + Chroma + RetrievalQA。

- **網路搜尋（Web Search Tool）**  
  - 工具名稱：`網路搜尋`
  - 功能：透過 Google Serper API 查詢網路最新醫療資訊。
  - 實作：`smart_web_search` 函數，包裝 Serper API。

- **Agent 整合**  
  - 使用 LangChain 的 `initialize_agent`，將上述兩個工具註冊給 Agent，並指定 AgentType 為 `ZERO_SHOT_REACT_DESCRIPTION`，讓 LLM 能根據描述自主選擇工具。

---

## 主要檔案說明

| 檔案名稱                | 用途說明                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| `react_agent_api.py`    | FastAPI 主後端，ReAct Agent 問答 API，整合 RAG 與 Web Search 工具         |
| `build_doctor_db.py`    | 將 `doctors.json` 轉為向量資料庫（Chroma），建立本地醫師知識庫            |
| `doctors.json`          | 醫師原始資料（姓名、專長、學歷、經歷等）                                 |
| `chroma_db/`            | Chroma 向量資料庫目錄                                                    |
| `client/`               | 前端專案（可選，React/Vite）                                             |
| `test_*.py`/`test_*.js` | 各種測試檔案，驗證 RAG、Web Search、API 等功能                            |

---

## 安裝與操作步驟

### 1. Python 環境安裝

建議使用 Python 3.9 以上版本。  
可用 Anaconda 或 venv 建立虛擬環境：

```bash
conda create -n myenv python=3.10
conda activate myenv
# 或
python -m venv myenv
myenv\Scripts\activate
```

---

### 2. 取得 API 金鑰

- **OpenAI API Key**：用於 GPT-4o LLM  
  申請：[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- **Serper API Key**：用於 Google 搜尋  
  申請：[https://serper.dev/](https://serper.dev/)

---

### 3. 設定環境變數

在專案根目錄建立 `.env` 檔案，內容如下：

```env
OPENAI_API_KEY=你的OpenAI金鑰
SERPER_API_KEY=你的Serper金鑰
```

---

### 4. 安裝依賴套件

```bash
pip install -r requirements.txt
```

**requirements.txt 範例：**
```txt
fastapi
uvicorn
langchain
langchain-community
langchain-huggingface
langchain-openai
chromadb
python-dotenv
sentence-transformers
transformers
torch
duckdb
requests
pytest
```

---

### 5. 建立本地向量資料庫

```bash
python build_doctor_db.py
```
- 此步驟會根據 `doctors.json` 產生 `chroma_db/` 目錄。

---

### 6. 啟動 FastAPI 後端

```bash
python react_agent_api.py
```
- 啟動後，API 端點為：`http://localhost:8000/api/ask`

---

### 7. 前端啟動（可選）

若有前端（如 React/Vite），進入 `client/` 目錄：

```bash
cd client
npm install
npm run dev
```
- 前端預設在 `http://localhost:5173`，可自行修改。

---

### 8. 測試

- 執行測試檔案（以 Python 為例）：
  ```bash
  python test_agent_tool.py
  ```
- 也可用 Postman/curl 測試 API。

---

## API 說明

### 問答 API

- **端點**：`POST /api/ask`
- **參數**：JSON 格式，`{"question": "你的問題"}`
- **回傳**：`{"result": "AI 回答內容"}`

#### 範例

```bash
curl -X POST "http://localhost:8000/api/ask" -H "Content-Type: application/json" -d "{\"question\": \"林宗翰醫師的專長\"}"
```

---

## 前端串接範例

假設你用 React + Axios：

```js
import axios from "axios";

async function askAgent(question) {
  const response = await axios.post("http://localhost:8000/api/ask", {
    question,
  });
  return response.data.result;
}
```

---

## 技術棧與設計邏輯

- **FastAPI**：高效 Python Web 框架，提供 RESTful API。
- **LangChain**：負責 Agent、RAG、工具鏈整合。
- **HuggingFace Embeddings**：文本向量化（`BAAI/bge-m3`）。
- **Chroma**：本地向量資料庫，支援高效檢索。
- **Google Serper API**：網路搜尋，取得最新醫療資訊。
- **OpenAI GPT-4o**：作為 LLM，負責推理、整合與生成答案。
- **Uvicorn**：ASGI 伺服器，啟動 FastAPI。

### 設計邏輯

1. 啟動時載入本地醫師向量資料庫與嵌入模型。
2. 註冊兩個 LangChain Tool（RAG、本地查詢 + Web Search）。
3. 用戶發問時，ReAct Agent 會根據問題內容自動選擇工具，並可多次交互推理。
4. 最終由 LLM 統整所有查詢結果，生成專業且自然的回覆。

---

## 常見問題排查

- **模型下載很慢或卡住**：請確認網路暢通，或手動下載 HuggingFace 模型到本地並指定路徑。
- **API Key 無效**：請確認 `.env` 檔案內容正確，且金鑰未過期。
- **Chroma 警告**：如遇到 Chroma 棄用警告，請參考官方文檔升級寫法。
- **推送 GitHub 出錯**：請確認已經有 commit，且分支名稱為 `main`。
- **前端 CORS 問題**：FastAPI 已預設允許所有來源跨域，如有特殊需求可調整 `CORSMiddleware` 設定。

---

## 聯絡方式

如有問題，歡迎開 issue 或聯絡作者。

---
