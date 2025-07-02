#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time

def test_rag_integration():
    """測試 RAG 整合功能"""
    print("🧪 測試 RAG + 搜尋整合系統")
    print("=" * 50)
    
    # 測試查詢
    test_queries = [
        "心臟病專家推薦",
        "糖尿病治療醫師",
        "高血壓專科醫師"
    ]
    
    base_url = "http://localhost:3001"
    
    for query in test_queries:
        print(f"\n🔍 測試查詢: {query}")
        print("-" * 40)
        
        try:
            # 發送查詢請求
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 查詢成功")
                
                # 顯示 RAG 結果
                if 'ragResults' in result and result['ragResults'].get('success'):
                    rag_count = result['ragResults'].get('count', 0)
                    print(f"📊 RAG 檢索: 找到 {rag_count} 位相關醫師")
                    
                    for i, doctor in enumerate(result['ragResults'].get('results', [])):
                        print(f"  {i+1}. {doctor['name']} - {doctor['specialty']}")
                else:
                    print("❌ RAG 檢索: 沒有找到相關醫師")
                
                # 顯示搜尋結果
                if 'searchResults' in result and result['searchResults'].get('organic'):
                    search_count = len(result['searchResults']['organic'])
                    print(f"🌐 網路搜尋: 找到 {search_count} 個結果")
                else:
                    print("❌ 網路搜尋: 沒有找到結果")
                
                # 顯示 AI 回答
                print(f"\n🤖 AI 回答:")
                print(result.get('response', '無回答'))
                
            else:
                print(f"❌ 查詢失敗: HTTP {response.status_code}")
                print(f"錯誤訊息: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ 無法連接到後端服務器")
            print("請確保後端服務器在 http://localhost:3001 運行")
            break
        except requests.exceptions.Timeout:
            print("❌ 查詢超時")
        except Exception as e:
            print(f"❌ 查詢錯誤: {str(e)}")
        
        print("\n" + "=" * 60)
        time.sleep(1)  # 避免請求過於頻繁

if __name__ == "__main__":
    test_rag_integration() 