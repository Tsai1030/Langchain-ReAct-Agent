import chromadb
from sentence_transformers import SentenceTransformer

query = "高醫心臟血管內科主治醫師有哪些專長？"
model = SentenceTransformer('BAAI/bge-m3')
client = chromadb.PersistentClient(path="chroma_db/doctorv1")
collection = client.get_or_create_collection("doctorv1")

query_vec = model.encode(query)
results = collection.query(query_embeddings=[query_vec.tolist()], n_results=3)
print("查詢結果：")
for doc in results['documents'][0]:
    print(doc)