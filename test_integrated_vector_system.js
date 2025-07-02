const { processMedicalQuery } = require('./server/services/queryService');

// 測試整合的向量 RAG 系統
async function testIntegratedVectorSystem() {
  console.log('🚀 開始測試整合的向量 RAG 系統...\n');

  const testQueries = [
    '心臟科醫師推薦',
    '內科主任',
    '糖尿病專科醫師',
    '高醫內科現在看到幾號',
    '腎臟科醫師'
  ];

  for (let i = 0; i < testQueries.length; i++) {
    const query = testQueries[i];
    console.log(`\n📝 測試查詢 ${i + 1}: "${query}"`);
    console.log('=' .repeat(50));

    try {
      const startTime = Date.now();
      const result = await processMedicalQuery(query);
      const endTime = Date.now();
      const duration = endTime - startTime;

      console.log(`⏱️ 查詢耗時: ${duration}ms`);
      console.log(`📊 資料來源: ${result.dataSource}`);
      
      // 顯示 RAG 結果統計
      if (result.ragResults.success) {
        console.log(`🔍 RAG 檢索成功: ${result.ragResults.count} 位醫師`);
        console.log(`🔧 使用的方法: ${result.ragResults.methods.join(', ')}`);
      } else {
        console.log('❌ RAG 檢索未找到相關醫師');
      }

      // 顯示向量 RAG 結果
      if (result.vectorRagResults.success) {
        console.log(`🧠 向量 RAG: ${result.vectorRagResults.count} 位醫師`);
      } else {
        console.log('❌ 向量 RAG 未找到相關醫師');
      }

      // 顯示關鍵字 RAG 結果
      if (result.keywordRagResults.success) {
        console.log(`🔤 關鍵字 RAG: ${result.keywordRagResults.count} 位醫師`);
      } else {
        console.log('❌ 關鍵字 RAG 未找到相關醫師');
      }

      // 顯示搜尋結果
      if (result.searchResults.realTimeData) {
        console.log('📡 取得即時叫號資訊');
      } else if (result.searchResults.organic && result.searchResults.organic.length > 0) {
        console.log(`🌐 網路搜尋: ${result.searchResults.organic.length} 個結果`);
      }

      console.log('\n💬 AI 回應:');
      console.log(result.response);
      console.log('\n' + '=' .repeat(50));

    } catch (error) {
      console.error(`❌ 查詢失敗: ${error.message}`);
      console.log('\n' + '=' .repeat(50));
    }

    // 等待一下再進行下一個查詢
    if (i < testQueries.length - 1) {
      console.log('⏳ 等待 2 秒...\n');
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  console.log('\n✅ 整合向量 RAG 系統測試完成！');
}

// 執行測試
if (require.main === module) {
  testIntegratedVectorSystem().catch(console.error);
}

module.exports = { testIntegratedVectorSystem }; 