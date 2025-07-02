const VectorRAGService = require('./server/services/vectorRagService');

// 測試向量 RAG 服務
async function testVectorRAGOnly() {
  console.log('🧠 測試向量 RAG 服務...\n');

  const vectorRagService = new VectorRAGService();

  const testQueries = [
    '心臟科醫師',
    '高血壓專家',
    '朱志生醫師'
  ];

  for (let i = 0; i < testQueries.length; i++) {
    const query = testQueries[i];
    console.log(`\n📝 測試查詢 ${i + 1}: "${query}"`);
    console.log('=' .repeat(40));

    try {
      const result = await vectorRagService.searchDoctors(query);
      console.log(`📊 檢索結果: ${result.success ? '成功' : '失敗'}`);
      
      if (result.success) {
        console.log(`🔍 找到 ${result.count} 位相關醫師`);
      } else {
        console.log('❌ 未找到相關醫師');
      }

    } catch (error) {
      console.error(`❌ 查詢失敗: ${error.message}`);
    }

    console.log('\n' + '=' .repeat(40));
  }

  console.log('\n✅ 向量 RAG 服務測試完成！');
}

// 執行測試
if (require.main === module) {
  testVectorRAGOnly().catch(console.error);
}

module.exports = { testVectorRAGOnly }; 