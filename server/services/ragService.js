const { spawn } = require('child_process');
const path = require('path');

// RAG 服務 - 整合 ChromaDB 檢索
class RAGService {
  constructor() {
    this.pythonPath = 'python'; // 預設使用系統 Python
    this.ragScriptPath = path.join(__dirname, '../../rag_with_gpt.py');
  }

  // 檢查是否為醫師查詢
  isDoctorQuery(query) {
    const doctorKeywords = [
      '醫師', '醫生', '專科', '專長', '治療', '診斷', '心臟', '內科', '外科',
      '婦產科', '小兒科', '眼科', '耳鼻喉科', '骨科', '泌尿科', '皮膚科',
      '神經科', '精神科', '腫瘤科', '復健科', '急診科', '家庭醫學科',
      'doctor', 'specialist', 'cardiology', 'internal medicine', 'surgery'
    ];
    
    return doctorKeywords.some(keyword => 
      query.toLowerCase().includes(keyword.toLowerCase())
    );
  }

  // 執行 RAG 檢索
  async performRAGSearch(query) {
    return new Promise((resolve, reject) => {
      console.log('🔍 執行 RAG 檢索...');
      
      // 建立一個獨立的 Python 腳本來執行 RAG 檢索
      const pythonScript = `
import sys
import os
import json

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    # 初始化 ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("doctorv1")
    
    # 初始化 embedding 模型
    model = SentenceTransformer("BAAI/bge-m3")
    
    # 執行檢索
    query_embedding = model.encode("${query.replace(/"/g, '\\"')}")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    
    # 格式化結果
    formatted_results = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        formatted_results.append({
            name: metadata['name'],
            department: metadata['department'],
            specialty: metadata['specialty'],
            title: metadata['title'],
            relevance: results['distances'][0][i],
            description: doc
        })
    
    print(json.dumps({
        'success': True,
        'results': formatted_results,
        'count': len(formatted_results)
    }))
    
except Exception as e:
    print(json.dumps({
        'success': False,
        'error': str(e),
        'traceback': str(sys.exc_info())
    }))
`;

      // 建立臨時 Python 檔案
      const tempScriptPath = path.join(__dirname, '../temp_rag_search.py');
      const fs = require('fs');
      
      try {
        fs.writeFileSync(tempScriptPath, pythonScript);
        
        const pythonProcess = spawn(this.pythonPath, [tempScriptPath], {
          cwd: path.join(__dirname, '../..'),
          stdio: ['pipe', 'pipe', 'pipe']
        });

        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
          output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
          errorOutput += data.toString();
        });

        pythonProcess.on('close', (code) => {
          // 清理臨時檔案
          try {
            fs.unlinkSync(tempScriptPath);
          } catch (e) {
            // 忽略清理錯誤
          }
          
          if (code === 0 && output.trim()) {
            try {
              const result = JSON.parse(output.trim());
              if (result.success) {
                console.log(`✅ RAG 檢索完成，找到 ${result.count} 位相關醫師`);
                resolve(result);
              } else {
                console.log(`❌ RAG 檢索失敗: ${result.error}`);
                resolve({ success: false, results: [], count: 0 });
              }
            } catch (e) {
              console.log(`❌ RAG 結果解析失敗: ${e.message}`);
              console.log(`原始輸出: ${output}`);
              resolve({ success: false, results: [], count: 0 });
            }
          } else {
            console.log(`❌ RAG 檢索執行失敗 (code: ${code})`);
            console.log(`錯誤輸出: ${errorOutput}`);
            console.log(`標準輸出: ${output}`);
            resolve({ success: false, results: [], count: 0 });
          }
        });

        pythonProcess.on('error', (error) => {
          console.log(`❌ RAG 檢索啟動失敗: ${error.message}`);
          resolve({ success: false, results: [], count: 0 });
        });
        
      } catch (error) {
        console.log(`❌ 建立臨時腳本失敗: ${error.message}`);
        resolve({ success: false, results: [], count: 0 });
      }
    });
  }

  // 整合 RAG 結果到搜尋結果中
  integrateRAGResults(ragResults, searchResults) {
    if (!ragResults.success || ragResults.count === 0) {
      return searchResults;
    }

    // 建立 RAG 結果摘要
    const ragSummary = ragResults.results.map((doctor, index) => {
      return `${index + 1}. ${doctor.name} - ${doctor.specialty} (${doctor.title})
   相關度: ${doctor.relevance.toFixed(3)}
   專長: ${doctor.specialty}`;
    }).join('\n\n');

    // 將 RAG 結果加入到搜尋結果中
    const enhancedResults = {
      ...searchResults,
      ragResults: ragResults.results,
      ragSummary: ragSummary,
      hasRAGData: true
    };

    return enhancedResults;
  }

  // 建立整合的 prompt
  createIntegratedPrompt(query, ragResults, searchResults) {
    let prompt = `您是一個專業的醫療資訊查詢助手。請根據以下資訊回答使用者的問題。

使用者問題: "${query}"

`;

    // 如果有即時資訊，優先顯示
    if (searchResults && searchResults.realTimeData) {
      const realTimeData = searchResults.realTimeData;
      prompt += `即時叫號資訊:
- 醫院: ${realTimeData.hospital || '高醫'}
- 科別: ${realTimeData.department || '無法取得'}
- 醫師: ${realTimeData.doctor || '無法取得'}
- 當前號碼: ${realTimeData.currentNumber || '無法取得'}
- 更新時間: ${realTimeData.timestamp ? new Date(realTimeData.timestamp).toLocaleString('zh-TW') : '無法取得'}

`;
    }

    // 如果有 RAG 結果，顯示醫師資訊
    if (ragResults.success && ragResults.count > 0) {
      prompt += `相關醫師資訊:
${ragResults.results.map((doctor, index) => 
  `${index + 1}. ${doctor.name} - ${doctor.specialty} (${doctor.title})`
).join('\n')}

`;
    }

    // 如果有網路搜尋結果，也加入
    if (searchResults && searchResults.organic && searchResults.organic.length > 0) {
      prompt += `網路搜尋結果:
${searchResults.organic.slice(0, 3).map((result, index) => 
  `${index + 1}. ${result.title}
   摘要: ${result.snippet || '無摘要'}`
).join('\n\n')}

`;
    }

    prompt += `請根據以上資訊提供準確、有用的回答。

回答要求:
1. 使用繁體中文回答
2. 如果有即時叫號資訊，請優先回答叫號進度
3. 如果有相關醫師資訊，可以補充醫師背景
4. 結合網路搜尋結果提供完整資訊
5. 回答要簡潔明瞭
6. 如果涉及醫院叫號資訊，請特別標註時間
7. 回答長度控制在 300 字以內

請直接回答，不要包含任何前綴或格式說明。`;

    return prompt;
  }
}

module.exports = RAGService; 