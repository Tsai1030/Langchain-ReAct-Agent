#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_rag_search():
    """測試 RAG 搜尋功能"""
    print("🧪 測試 RAG 搜尋功能")
    print("=" * 50)
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        print("✅ 成功導入必要套件")
        
        # 初始化 ChromaDB
        print("🔧 初始化 ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("doctorv1")
        print("✅ ChromaDB 初始化成功")
        
        # 初始化 embedding 模型
        print("🤖 載入 BGE-M3 模型...")
        model = SentenceTransformer("BAAI/bge-m3")
        print("✅ 模型載入成功")
        
        # 測試查詢
        test_query = "心臟病專家"
        print(f"🔍 測試查詢: {test_query}")
        
        # 執行檢索
        query_embedding = model.encode(test_query)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        print(f"✅ 檢索成功，找到 {len(results['documents'][0])} 個結果")
        
        # 顯示結果
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"{i+1}. {metadata['name']} - {metadata['specialty']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_search()
    if success:
        print("\n✅ RAG 搜尋功能正常")
    else:
        print("\n❌ RAG 搜尋功能有問題") 