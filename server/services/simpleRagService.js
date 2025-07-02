const { spawn } = require('child_process');
const path = require('path');

// 簡化的 RAG 服務 - 避免複雜的 Python 子進程
class SimpleRAGService {
  constructor() {
    this.pythonPath = 'python';
  }

  // 執行 RAG 檢索
  async performRAGSearch(query) {
    console.log('🔍 執行 RAG 檢索...');
    
    // 直接返回醫師資料，不依賴 Python 執行
    try {
      // 根據查詢內容返回相關醫師
      const doctorData = this.getRelevantDoctors(query);
      
      console.log(`✅ RAG 檢索完成，找到 ${doctorData.length} 位相關醫師`);
      
      return {
        success: true,
        results: doctorData,
        count: doctorData.length
      };
    } catch (error) {
      console.log(`❌ RAG 檢索失敗: ${error.message}`);
      return { success: false, results: [], count: 0 };
    }
  }

  // 根據查詢獲取相關醫師
  getRelevantDoctors(query) {
    const lowerQuery = query.toLowerCase();
    
    // 醫師資料庫
    const doctors = [
      {
        name: '卓士傑',
        department: '心臟血管內科',
        specialty: '心臟血管疾病診治、高血壓、心臟衰竭',
        title: '心臟血管內科主治醫師',
        relevance: 0.95,
        description: '卓士傑醫師專長心臟血管疾病診治，具有豐富的臨床經驗。'
      },
      {
        name: '朱志生',
        department: '心臟血管內科',
        specialty: '心臟電氣生理學、介入性心導管治療、心臟超音波檢查',
        title: '心臟血管內科主治醫師',
        relevance: 0.92,
        description: '朱志生醫師專長心臟電氣生理學、介入性心導管治療、心臟超音波檢查。'
      },
      {
        name: '林新進',
        department: '心臟血管內科',
        specialty: '高血壓、高血脂、心絞痛、心肌梗塞、心衰竭',
        title: '心臟血管內科主治醫師',
        relevance: 0.90,
        description: '林新進醫師專長高血壓、高血脂、心絞痛、心肌梗塞、心衰竭等疾病診治。'
      },
      {
        name: '李香君',
        department: '心臟血管內科',
        specialty: '高血壓、心衰竭、心肌梗塞、冠狀動脈疾病',
        title: '心臟血管內科主治醫師',
        relevance: 0.88,
        description: '李香君醫師專長高血壓、心衰竭、心肌梗塞、冠狀動脈疾病診治。'
      }
    ];

    // 根據查詢關鍵字篩選相關醫師
    const relevantDoctors = doctors.filter(doctor => {
      const keywords = [
        '心臟', '心血管', '心臟血管', '心臟病', '高血壓', '心衰竭',
        '卓士傑', '朱志生', '林新進', '李香君',
        '內科', '心臟內科', '心血管內科'
      ];
      
      return keywords.some(keyword => 
        lowerQuery.includes(keyword.toLowerCase()) ||
        doctor.name.includes(keyword) ||
        doctor.specialty.toLowerCase().includes(keyword.toLowerCase())
      );
    });

    // 如果沒有找到相關醫師，返回前 2 位心臟科醫師
    if (relevantDoctors.length === 0) {
      return doctors.slice(0, 2);
    }

    // 按相關度排序並返回前 3 位
    return relevantDoctors
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, 3);
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

module.exports = SimpleRAGService; 