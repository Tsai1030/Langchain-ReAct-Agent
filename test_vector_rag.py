# -*- coding: utf-8 -*-
import sys
import os

# 設定 UTF-8 編碼
if sys.platform.startswith('win'):
    # Windows 環境下設定控制台編碼
    os.system('chcp 65001 > nul')
    # 重新設定 stdout 和 stderr 編碼
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

import chromadb
from sentence_transformers import SentenceTransformer
import json

def test_vector_rag():
    print("🧪 測試向量 RAG 檢索功能")
    print("=" * 50)
    
    try:
        # 1. 初始化模型和資料庫
        print("🔧 初始化模型和資料庫...")
        model = SentenceTransformer('BAAI/bge-m3')
        
        # <<< 主要修改點：路徑應指向父目錄 'chroma_db'
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection("doctorv1")
        
        # 2. 載入原始醫師資料（用於比對）
        with open('doctors.json', 'r', encoding='utf-8') as f:
            doctors = json.load(f)
        
        db_count = collection.count()
        if db_count == 0:
            print("⚠️ 警告: 資料庫是空的！請先執行 build_doctor_db.py 建立資料庫。")
            return False

        print(f"✅ 成功載入 {len(doctors)} 位醫師資料，資料庫中有 {db_count} 筆向量。")
        
        # 3. 測試查詢案例
        test_queries = [
            {
                "query": "高醫朱志生醫師的專長是什麼？",
                "expected": "朱志生",
                "description": "醫師姓名查詢"
            },
            {
                "query": "心臟電氣生理學專家",
                "expected": "心臟電氣生理學",
                "description": "專長關鍵字查詢"
            },
            {
                "query": "高血壓治療醫師",
                "expected": "高血壓",
                "description": "疾病關鍵字查詢"
            },
            {
                "query": "高雄醫學大學心臟血管內科主治醫師",
                "expected": "心臟血管內科",
                "description": "科別和職稱查詢"
            },
            {
                "query": "介入性心導管治療專家",
                "expected": "介入性心導管治療",
                "description": "治療技術查詢"
            }
        ]
        
        print(f"\n🔍 開始測試 {len(test_queries)} 個查詢案例...")
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n 測試 {i}: {test_case['description']}")
            print(f"查詢: \"{test_case['query']}\"")
            
            # 執行向量檢索
            query_vec = model.encode(test_case['query'])
            results = collection.query(
                query_embeddings=[query_vec.tolist()], 
                n_results=3
            )
            
            # 顯示結果
            print("🔍 檢索結果:")
            for j, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                print(f"  {j+1}. 相似度: {1-distance:.3f}")
                print(f"     內容: {doc[:100]}...")
                
                # 檢查是否包含預期內容
                if test_case['expected'].lower() in doc.lower():
                    print(f"     ✅ 包含預期內容: {test_case['expected']}")
                else:
                    print(f"     ⚠️ 未包含預期內容: {test_case['expected']}")
            
            print("-" * 40)
        
        # 4. 測試資料庫統計
        print("\n📊 資料庫統計:")
        print(f"總醫師數量: {len(doctors)}")
        print(f"向量資料庫大小: {db_count} 筆")
        
        # 5. 測試相似度排序
        print("\n 測試相似度排序:")
        query = "心臟血管疾病專家"
        query_vec = model.encode(query)
        results = collection.query(
            query_embeddings=[query_vec.tolist()], 
            n_results=5
        )
        
        print(f"查詢: \"{query}\"")
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            similarity = 1 - distance
            print(f"  {i+1}. 相似度: {similarity:.3f}")
            # 提取醫師姓名
            doctor_name = doc.split('，')[0] if '，' in doc else doc[:10]
            print(f"     醫師: {doctor_name}")
        
        print("\n✅ 向量 RAG 測試完成！")
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ 測試失敗: {str(e)}")
        traceback.print_exc() # 印出更詳細的錯誤追蹤
        return False

def test_single_query(query):
    """測試單一查詢"""
    print(f" 測試單一查詢: \"{query}\"")
    print("-" * 40)
    
    try:
        model = SentenceTransformer('BAAI/bge-m3')
        # 這裡的路徑已是正確的，無需修改
        client = chromadb.PersistentClient(path="chroma_db") 
        collection = client.get_or_create_collection("doctorv1")
        
        query_vec = model.encode(query)
        results = collection.query(
            query_embeddings=[query_vec.tolist()], 
            n_results=3
        )
        
        print("🔍 檢索結果:")
        for i, (doc, distance, metadata) in enumerate(zip(results['documents'][0], results['distances'][0], results['metadatas'][0])):
            similarity = 1 - distance
            print(f"\n{i+1}. 相似度: {similarity:.3f}")
            print(f"醫師姓名: {metadata.get('name', 'N/A')}")
            print(f"科別: {metadata.get('department', 'N/A')}")
            print(f"專長: {'、'.join(metadata.get('specialty', []))}")
            # print(f"完整內容: {doc}") # 可以選擇性印出完整 document
            
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ 查詢失敗: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 測試單一查詢
        query = " ".join(sys.argv[1:])
        test_single_query(query)
    else:
        # 執行完整測試
        test_vector_rag()