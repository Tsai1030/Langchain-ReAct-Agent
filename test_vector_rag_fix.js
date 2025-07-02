const VectorRAGService = require('./server/services/vectorRagService');

async function testVectorRAG() {
  console.log('🧪 測試向量 RAG 修復...\n');
  
  const vectorRagService = new VectorRAGService();
  
  const testQueries = [
    '心臟血管內科',
    '卓士傑醫師',
    '高雄醫學大學附設醫院心臟科'
  ];
  
  for (const query of testQueries) {
    console.log(`\n🔍 測試查詢: "${query}"`);
    console.log('─'.repeat(50));
    
    try {
      const result = await vectorRagService.searchDoctors(query);
      
      if (result.success) {
        console.log(`✅ 成功找到 ${result.count} 位醫師`);
        if (result.results.length > 0) {
          console.log('📋 醫師資訊:');
          result.results.forEach((doctor, index) => {
            console.log(`  ${index + 1}. ${doctor.name} - ${doctor.department}`);
            if (doctor.specialty && doctor.specialty.length > 0) {
              console.log(`     專長: ${doctor.specialty.join(', ')}`);
            }
            console.log(`     相關度: ${doctor.relevance.toFixed(3)}`);
          });
        }
      } else {
        console.log('❌ 檢索失敗');
      }
    } catch (error) {
      console.log(`❌ 執行錯誤: ${error.message}`);
    }
  }
  
  console.log('\n🎯 測試完成！');
}

// 執行測試
testVectorRAG().catch(console.error); 