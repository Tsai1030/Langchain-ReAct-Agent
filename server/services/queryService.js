const axios = require('axios');
const OpenAI = require('openai');
const ScrapingBeeService = require('./scrapingBeeService');
const DoctorRAGService = require('./doctorRagService');
const VectorRAGService = require('./vectorRagService');

// 檢查環境變數
if (!process.env.OPENAI_API_KEY) {
  console.error('❌ 錯誤：OPENAI_API_KEY 環境變數未設定');
  console.error('請在 server/.env 檔案中設定您的 OpenAI API 金鑰');
  console.error('範例：OPENAI_API_KEY=sk-your-api-key-here');
}

if (!process.env.SCRAPING_BEE_KEY) {
  console.warn('⚠️ 警告：SCRAPING_BEE_KEY 環境變數未設定');
  console.warn('動態網頁搜尋功能將被停用');
}

if (!process.env.SERPER_API_KEY) {
  console.warn('⚠️ 警告：SERPER_API_KEY 環境變數未設定');
  console.warn('Google 搜尋功能將被停用，將使用 ScrapingBee 作為備用');
}

// 初始化 OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'dummy-key',
});

// Serper API 搜尋服務
class SerperSearchService {
  constructor() {
    this.apiKey = process.env.SERPER_API_KEY;
    this.baseURL = 'https://google.serper.dev/search';
  }

  async search(query) {
    try {
      if (!this.apiKey) {
        console.log('⚠️ Serper API key 未設定，跳過 Google 搜尋');
        return { organic: [], totalResults: 0 };
      }

      console.log(`🔍 執行 Google 搜尋: ${query}`);

      const response = await axios.post(this.baseURL, {
        q: query,
        num: 10, // 取得前 10 個結果
        gl: 'tw', // 台灣地區
        hl: 'zh-tw' // 中文繁體
      }, {
        headers: {
          'X-API-KEY': this.apiKey,
          'Content-Type': 'application/json'
        },
        timeout: 10000 // 10 秒超時
      });

      return response.data;
    } catch (error) {
      console.error('Serper 搜尋錯誤:', error.message);
      // 返回空結果而不是拋出錯誤
      return { organic: [], totalResults: 0 };
    }
  }
}

// 整合 GPT 分析服務
class IntegratedGPTAnalysisService {
  async analyzeQuery(query, ragResults, searchResults) {
    try {
      if (!process.env.OPENAI_API_KEY) {
        throw new Error('OpenAI API key 未設定，請在 server/.env 檔案中設定 OPENAI_API_KEY');
      }

      console.log('🤖 使用 GPT-4o 分析整合結果');

      // 建立整合的 prompt
      const prompt = this.createIntegratedPrompt(query, ragResults, searchResults);

      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "您是一個專業的醫療資訊查詢助手，專門協助使用者查詢醫院相關資訊和醫師推薦。"
          },
          {
            role: "user",
            content: prompt
          }
        ],
        max_tokens: 800,
        temperature: 0.3
      });

      return completion.choices[0].message.content.trim();
    } catch (error) {
      console.error('GPT 分析錯誤:', error.message);
      throw new Error(`AI 分析錯誤: ${error.message}`);
    }
  }

  // 建立整合的 prompt
  createIntegratedPrompt(query, ragResults, searchResults) {
    let prompt = `您是一個專業的醫療資訊查詢助手。請根據以下資訊回答使用者的問題。

使用者問題: "${query}"

`;

    // 如果有醫師資訊，優先顯示
    if (ragResults && ragResults.success && ragResults.count > 0) {
      prompt += `相關醫師資訊:
${ragResults.results.map((doctor, index) => 
  `${index + 1}. ${doctor.name} - ${doctor.department}
   專長: ${doctor.specialty ? doctor.specialty.join(', ') : '無資料'}
   職稱: ${doctor.title ? doctor.title[0] : '無資料'}
   相關度: ${doctor.relevance.toFixed(3)}`
).join('\n\n')}

`;
    }

    // 如果有即時資訊，顯示叫號進度
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
2. 如果有醫師資訊，請詳細介紹醫師背景和專長
3. 如果有即時叫號資訊，請優先回答叫號進度
4. 結合網路搜尋結果提供完整資訊
5. 回答要簡潔明瞭，結構清晰
6. 如果涉及醫院叫號資訊，請特別標註時間
7. 回答長度控制在 400 字以內

請直接回答，不要包含任何前綴或格式說明。`;

    return prompt;
  }
}

// 主要查詢處理服務
class MedicalQueryService {
  constructor() {
    this.searchService = new SerperSearchService();
    this.analysisService = new IntegratedGPTAnalysisService();
    this.scrapingBeeService = new ScrapingBeeService();
    this.ragService = new DoctorRAGService();
    this.vectorRagService = new VectorRAGService();
  }

  async processMedicalQuery(query) {
    try {
      console.log(`📝 開始處理查詢: ${query}`);

      // 1. 並行執行三種檢索：關鍵字 RAG、向量 RAG 和 Web 搜尋
      console.log('🔍 步驟 1: 並行執行關鍵字 RAG、向量 RAG 和 Web 搜尋...');
      
      const [keywordRagResults, vectorRagResults, searchResults] = await Promise.allSettled([
        this.ragService.searchDoctors(query),
        this.vectorRagService.searchDoctors(query),
        this.searchService.search(this.optimizeSearchQuery(query))
      ]);

      // 處理關鍵字 RAG 結果
      const keywordRagData = keywordRagResults.status === 'fulfilled' ? keywordRagResults.value : { success: false, results: [], count: 0 };
      
      // 處理向量 RAG 結果
      const vectorRagData = vectorRagResults.status === 'fulfilled' ? vectorRagResults.value : { success: false, results: [], count: 0 };
      
      // 處理搜尋結果
      const searchData = searchResults.status === 'fulfilled' ? searchResults.value : { organic: [], totalResults: 0 };
      
      // 2. 合併 RAG 結果（優先使用向量檢索結果）
      console.log('🔍 步驟 2: 合併 RAG 結果...');
      const mergedRagData = this.mergeRagResults(keywordRagData, vectorRagData);
      
      // 3. 檢查是否需要即時資訊
      console.log('🔍 步驟 3: 檢查即時資訊...');
      const hybridResult = await this.scrapingBeeService.hybridSearch(query, searchData);
      
      let response;
      let finalSearchResults;

      if (hybridResult.type === 'real-time') {
        // 有即時資訊，整合 RAG 結果
        console.log('✅ 使用即時資訊 + RAG 結果');
        finalSearchResults = {
          organic: [],
          totalResults: 0,
          realTimeData: hybridResult.data
        };
      } else {
        // 使用 Google 搜尋結果 + RAG 結果
        console.log('✅ 使用 Google 搜尋結果 + RAG 結果');
        finalSearchResults = {
          organic: searchData.organic?.slice(0, 3) || [],
          totalResults: searchData.searchInformation?.totalResults || 0
        };
      }

      // 4. 使用 GPT 整合所有結果
      console.log('🤖 步驟 4: 使用 GPT 整合結果...');
      response = await this.analysisService.analyzeQuery(query, mergedRagData, finalSearchResults);

      console.log(`✅ 查詢處理完成`);

      return {
        response,
        searchResults: finalSearchResults,
        ragResults: mergedRagData,
        keywordRagResults: keywordRagData,
        vectorRagResults: vectorRagData,
        dataSource: hybridResult.type
      };

    } catch (error) {
      console.error('查詢處理失敗:', error);
      throw error;
    }
  }

  // 合併關鍵字 RAG 和向量 RAG 結果
  mergeRagResults(keywordRagData, vectorRagData) {
    const mergedResults = {
      success: false,
      results: [],
      count: 0,
      methods: []
    };

    // 優先使用向量檢索結果（如果成功）
    if (vectorRagData.success && vectorRagData.count > 0) {
      mergedResults.success = true;
      mergedResults.results = vectorRagData.results;
      mergedResults.count = vectorRagData.count;
      mergedResults.methods.push('vector');
      console.log(`✅ 使用向量 RAG 結果: ${vectorRagData.count} 位醫師`);
    }
    // 如果向量檢索失敗，使用關鍵字檢索結果
    else if (keywordRagData.success && keywordRagData.count > 0) {
      mergedResults.success = true;
      mergedResults.results = keywordRagData.results;
      mergedResults.count = keywordRagData.count;
      mergedResults.methods.push('keyword');
      console.log(`✅ 使用關鍵字 RAG 結果: ${keywordRagData.count} 位醫師`);
    }
    // 如果兩種方法都失敗
    else {
      console.log('❌ 兩種 RAG 方法都未找到相關醫師');
    }

    return mergedResults;
  }

  // 格式化即時資訊回應
  formatRealTimeResponse(realTimeData) {
    console.log('formatRealTimeResponse:', realTimeData);
    const { hospital, currentNumber, department } = realTimeData;
    // 強制取得 timestamp
    let timestamp = realTimeData.timestamp;
    if (!timestamp) {
      timestamp = new Date().toISOString();
    }
    let timeStr = '';
    if (timestamp) {
      timeStr = `- 更新時間：${new Date(timestamp).toLocaleString('zh-TW')}`;
    }
    return `根據 ${hospital} 即時系統顯示：

目前叫號進度：
- 當前號碼：${currentNumber && currentNumber !== '' ? currentNumber : '無法取得'}
- 科別：${department && department !== '' ? department : '無法取得'}
${timeStr ? timeStr + '\n' : ''}
請攜帶健保卡提前報到，實際叫號進度請以現場公告為準。`;
  }

  // 優化搜尋查詢
  optimizeSearchQuery(userQuery) {
    // 針對醫療查詢進行優化
    let optimizedQuery = userQuery;

    // 如果是醫院叫號查詢，加入特定關鍵字
    if (userQuery.includes('看到幾號') || userQuery.includes('叫號')) {
      optimizedQuery += ' 即時叫號 門診進度';
    }

    // 如果是醫師查詢，加入醫院名稱
    if (userQuery.includes('醫師') && !userQuery.includes('高醫')) {
      optimizedQuery += ' 高雄醫學大學附設醫院';
    }

    return optimizedQuery;
  }
}

// 導出服務實例
const medicalQueryService = new MedicalQueryService();

// 導出處理函數
async function processMedicalQuery(query) {
  return await medicalQueryService.processMedicalQuery(query);
}

module.exports = { MedicalQueryService, processMedicalQuery };