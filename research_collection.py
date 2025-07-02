import chromadb

# 指定你的 chroma_db 目錄
client = chromadb.PersistentClient(path="chroma_db")

# 取得所有 collection 的資訊
collections = client.list_collections()

# 印出所有 collection 的名稱
for col in collections:
    print(col.name)