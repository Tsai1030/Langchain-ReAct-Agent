#!/usr/bin/env node

const axios = require('axios');

// 測試 RAG 服務
async function testRAGService() {
  console.log('🧪 測試 RAG 服務...');
  
  try {
    const response = await axios.post('http://localhost:3001/api/query', {
      query: '現在高醫卓士傑醫師 心臟血管內科４診看到幾號了?'
    });
    
    console.log('✅ RAG 服務測試結果:');
    console.log('狀態碼:', response.status);
    console.log('回應時間:', response.headers['x-response-time'] || '未知');
    
    if (response.data && response.data.answer) {
      console.log('回答長度:', response.data.answer.length);
      console.log('回答預覽:', response.data.answer.substring(0, 200) + '...');
    }
    
    return true;
  } catch (error) {
    console.log('❌ RAG 服務測試失敗:', error.message);
    return false;
  }
}

// 測試即時資訊服務
async function testRealTimeService() {
  console.log('\n🧪 測試即時資訊服務...');
  
  try {
    const response = await axios.post('http://localhost:3001/api/query', {
      query: '高醫心臟內科現在叫到幾號'
    });
    
    console.log('✅ 即時資訊服務測試結果:');
    console.log('狀態碼:', response.status);
    
    if (response.data && response.data.answer) {
      console.log('回答長度:', response.data.answer.length);
      console.log('回答預覽:', response.data.answer.substring(0, 200) + '...');
      
      // 檢查是否包含即時資訊
      if (response.data.answer.includes('號') || response.data.answer.includes('即時')) {
        console.log('✅ 回答包含即時資訊');
      } else {
        console.log('⚠️ 回答可能不包含即時資訊');
      }
    }
    
    return true;
  } catch (error) {
    console.log('❌ 即時資訊服務測試失敗:', error.message);
    return false;
  }
}

// 測試整合查詢
async function testIntegratedQuery() {
  console.log('\n🧪 測試整合查詢...');
  
  try {
    const response = await axios.post('http://localhost:3001/api/query', {
      query: '高醫卓士傑醫師的專長是什麼？現在心臟內科叫到幾號？'
    });
    
    console.log('✅ 整合查詢測試結果:');
    console.log('狀態碼:', response.status);
    
    if (response.data && response.data.answer) {
      console.log('回答長度:', response.data.answer.length);
      console.log('完整回答:');
      console.log(response.data.answer);
    }
    
    return true;
  } catch (error) {
    console.log('❌ 整合查詢測試失敗:', error.message);
    return false;
  }
}

// 主測試函數
async function runTests() {
  console.log('🚀 開始系統測試...\n');
  
  const results = {
    rag: await testRAGService(),
    realtime: await testRealTimeService(),
    integrated: await testIntegratedQuery()
  };
  
  console.log('\n📊 測試結果總結:');
  console.log('RAG 服務:', results.rag ? '✅ 通過' : '❌ 失敗');
  console.log('即時資訊服務:', results.realtime ? '✅ 通過' : '❌ 失敗');
  console.log('整合查詢:', results.integrated ? '✅ 通過' : '❌ 失敗');
  
  const successCount = Object.values(results).filter(Boolean).length;
  console.log(`\n🎯 總體結果: ${successCount}/3 項測試通過`);
  
  if (successCount === 3) {
    console.log('🎉 所有測試通過！系統運行正常。');
  } else {
    console.log('⚠️ 部分測試失敗，請檢查相關服務。');
  }
}

// 執行測試
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { testRAGService, testRealTimeService, testIntegratedQuery, runTests }; 