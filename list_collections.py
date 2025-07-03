import chromadb

def list_all_collections():
    """列出ChromaDB中的所有集合"""
    try:
        # 連接到ChromaDB
        client = chromadb.PersistentClient(path="chroma_db")
        
        # 獲取所有集合
        collections = client.list_collections()
        
        print("=== ChromaDB 集合列表 ===")
        print(f"總共找到 {len(collections)} 個集合:")
        print()
        
        for i, collection in enumerate(collections, 1):
            print(f"{i}. 集合名稱: {collection.name}")
            print(f"   集合ID: {collection.id}")
            
            # 獲取集合中的文檔數量
            try:
                count = collection.count()
                print(f"   文檔數量: {count}")
            except Exception as e:
                print(f"   文檔數量: 無法獲取 ({e})")
            
            print()
        
        if not collections:
            print("沒有找到任何集合")
            
    except Exception as e:
        print(f"連接ChromaDB時發生錯誤: {e}")

if __name__ == "__main__":
    list_all_collections()