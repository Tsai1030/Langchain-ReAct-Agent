const DoctorRAGService = require('./server/services/doctorRagService');
const VectorRAGService = require('./server/services/vectorRagService');

// 測試 RAG 功能（不需要 API 金鑰）
async function testRAGOnly() {
  console.log('🧠 測試 RAG 功能（關鍵字 + 向量）...\n');

  const keywordRagService = new DoctorRAGService();
  const vectorRagService = new VectorRAGService();

  const testQueries = [
    '心臟科醫師',
    '高血壓專家',
    '朱志生醫師',
    '介入性治療',
    '內科主任'
  ];

  for (let i = 0; i < testQueries.length; i++) {
    const query = testQueries[i];
    console.log(`\n📝 測試查詢 ${i + 1}: "${query}"`);
    console.log('=' .repeat(50));

    try {
      // 並行執行兩種 RAG 檢索
      const [keywordResults, vectorResults] = await Promise.allSettled([
        keywordRagService.searchDoctors(query),
        vectorRagService.searchDoctors(query)
      ]);

      // 處理關鍵字 RAG 結果
      const keywordData = keywordResults.status === 'fulfilled' ? keywordResults.value : { success: false, results: [], count: 0 };
      
      // 處理向量 RAG 結果
      const vectorData = vectorResults.status === 'fulfilled' ? vectorResults.value : { success: false, results: [], count: 0 };

      console.log(`🔤 關鍵字 RAG: ${keywordData.success ? '成功' : '失敗'} (${keywordData.count} 位醫師)`);
      console.log(`🧠 向量 RAG: ${vectorData.success ? '成功' : '失敗'} (${vectorData.count} 位醫師)`);

      // 顯示醫師資訊
      if (vectorData.success && vectorData.count > 0) {
        console.log('\n👨‍⚕️ 向量檢索結果:');
        vectorData.results.forEach((doctor, index) => {
          console.log(`\n${index + 1}. ${doctor.name} - ${doctor.department}`);
          if (doctor.title && doctor.title.length > 0) {
            console.log(`   職稱: ${doctor.title.join(', ')}`);
          }
          if (doctor.specialty && doctor.specialty.length > 0) {
            console.log(`   專長: ${doctor.specialty.join(', ')}`);
          }
          console.log(`   相關度: ${doctor.relevance.toFixed(3)}`);
        });
      } else if (keywordData.success && keywordData.count > 0) {
        console.log('\n👨‍⚕️ 關鍵字檢索結果:');
        keywordData.results.forEach((doctor, index) => {
          console.log(`\n${index + 1}. ${doctor.name} - ${doctor.department}`);
          if (doctor.title && doctor.title.length > 0) {
            console.log(`   職稱: ${doctor.title.join(', ')}`);
          }
          if (doctor.specialty && doctor.specialty.length > 0) {
            console.log(`   專長: ${doctor.specialty.join(', ')}`);
          }
          console.log(`   相關度: ${doctor.relevance.toFixed(3)}`);
        });
      } else {
        console.log('❌ 兩種 RAG 方法都未找到相關醫師');
      }

    } catch (error) {
      console.error(`❌ 查詢失敗: ${error.message}`);
    }

    console.log('\n' + '=' .repeat(50));
  }

  console.log('\n✅ RAG 功能測試完成！');
}

// 執行測試
if (require.main === module) {
  testRAGOnly().catch(console.error);
}

module.exports = { testRAGOnly };