#!/usr/bin/env node

const { MedicalQueryService } = require('./server/services/queryService');

async function testIntegratedSystem() {
  console.log('🧪 測試整合 RAG + 搜尋系統');
  console.log('=' * 50);

  const queryService = new MedicalQueryService();

  // 測試查詢
  const testQueries = [
    '心臟病專家推薦',
    '高醫心臟內科卓士傑醫師現在看到幾號?',
    '糖尿病治療醫師',
    '高血壓專科醫師'
  ];

  for (const query of testQueries) {
    console.log(`\n🔍 測試查詢: ${query}`);
    console.log('-'.repeat(40));

    try {
      const result = await queryService.processMedicalQuery(query);
      
      console.log('✅ 查詢處理成功');
      console.log('\n📊 RAG 結果:');
      if (result.ragResults.success) {
        console.log(`找到 ${result.ragResults.count} 位相關醫師`);
        result.ragResults.results.forEach((doctor, index) => {
          console.log(`${index + 1}. ${doctor.name} - ${doctor.specialty}`);
        });
      } else {
        console.log('❌ 沒有找到相關醫師');
      }

      console.log('\n🌐 搜尋結果:');
      if (result.searchResults.organic && result.searchResults.organic.length > 0) {
        console.log(`找到 ${result.searchResults.organic.length} 個網路結果`);
      } else {
        console.log('❌ 沒有網路搜尋結果');
      }

      console.log('\n🤖 AI 回答:');
      console.log(result.response);
      console.log('\n' + '='.repeat(60));

    } catch (error) {
      console.error(`❌ 查詢失敗: ${error.message}`);
    }
  }
}

// 執行測試
if (require.main === module) {
  testIntegratedSystem().catch(console.error);
}

module.exports = { testIntegratedSystem }; 