const axios = require('axios');

// 測試 RAG + Web Search 整合系統
async function testRAGWebIntegration() {
  console.log('🧪 測試 RAG + Web Search 整合系統');
  console.log('='.repeat(50));

  const testCases = [
    {
      name: '醫師資訊查詢',
      query: '高醫朱志生醫師的專長是什麼？',
      expected: ['醫師資訊', '專長']
    },
    {
      name: '即時叫號查詢',
      query: '現在高醫心臟內科叫到幾號了？',
      expected: ['即時資訊', '叫號']
    },
    {
      name: '綜合查詢',
      query: '高醫林宗翰醫師現在心臟內科叫到幾號？',
      expected: ['醫師資訊', '即時資訊']
    },
    {
      name: '醫師背景查詢',
      query: '李香君醫師的學歷和經歷',
      expected: ['醫師資訊', '學歷', '經歷']
    }
  ];

  let successCount = 0;
  let totalTests = testCases.length;

  for (const testCase of testCases) {
    console.log(`\n🔍 測試: ${testCase.name}`);
    console.log(`查詢: "${testCase.query}"`);
    console.log('-'.repeat(40));

    try {
      const startTime = Date.now();
      
      const response = await axios.post('http://localhost:3001/api/query', {
        query: testCase.query
      });

      const endTime = Date.now();
      const responseTime = endTime - startTime;

      console.log(`✅ 狀態碼: ${response.status}`);
      console.log(`⏱️ 回應時間: ${responseTime}ms`);

      if (response.data && response.data.answer) {
        const answer = response.data.answer;
        console.log(`📝 回答長度: ${answer.length} 字`);
        console.log(`📄 回答預覽: ${answer.substring(0, 150)}...`);

        // 檢查是否包含預期內容
        const hasExpectedContent = testCase.expected.every(expected => 
          answer.toLowerCase().includes(expected.toLowerCase())
        );

        if (hasExpectedContent) {
          console.log('✅ 回答包含預期內容');
          successCount++;
        } else {
          console.log('⚠️ 回答可能缺少預期內容');
        }

        // 檢查 RAG 結果
        if (response.data.ragResults && response.data.ragResults.success) {
          console.log(`👨‍⚕️ RAG 找到 ${response.data.ragResults.count} 位醫師`);
        } else {
          console.log('❌ RAG 沒有找到醫師資訊');
        }

        // 檢查搜尋結果
        if (response.data.searchResults) {
          if (response.data.searchResults.realTimeData) {
            console.log('✅ 包含即時資訊');
          }
          if (response.data.searchResults.organic && response.data.searchResults.organic.length > 0) {
            console.log(`🌐 包含 ${response.data.searchResults.organic.length} 個網路結果`);
          }
        }

      } else {
        console.log('❌ 沒有收到有效回答');
      }

    } catch (error) {
      console.log(`❌ 測試失敗: ${error.message}`);
      if (error.response) {
        console.log(`狀態碼: ${error.response.status}`);
        console.log(`錯誤訊息: ${error.response.data?.message || '未知錯誤'}`);
      }
    }
  }

  // 測試結果總結
  console.log('\n' + '='.repeat(50));
  console.log('📊 測試結果總結:');
  console.log(`✅ 成功: ${successCount}/${totalTests}`);
  console.log(`❌ 失敗: ${totalTests - successCount}/${totalTests}`);
  console.log(`🎯 成功率: ${((successCount / totalTests) * 100).toFixed(1)}%`);

  if (successCount === totalTests) {
    console.log('\n🎉 所有測試通過！RAG + Web Search 整合系統運行正常。');
  } else {
    console.log('\n⚠️ 部分測試失敗，請檢查系統配置。');
  }
}

// 測試單一查詢
async function testSingleQuery(query) {
  console.log(`🧪 測試單一查詢: "${query}"`);
  console.log('-'.repeat(40));

  try {
    const response = await axios.post('http://localhost:3001/api/query', {
      query: query
    });

    console.log('✅ 查詢成功');
    console.log('📄 完整回答:');
    console.log(response.data.answer);
    
    // 顯示詳細資訊
    if (response.data.ragResults) {
      console.log('\n👨‍⚕️ RAG 結果:');
      console.log(`- 成功: ${response.data.ragResults.success}`);
      console.log(`- 醫師數量: ${response.data.ragResults.count}`);
    }

    if (response.data.searchResults) {
      console.log('\n🌐 搜尋結果:');
      console.log(`- 資料來源: ${response.data.dataSource}`);
      if (response.data.searchResults.realTimeData) {
        console.log(`- 即時資訊: ${JSON.stringify(response.data.searchResults.realTimeData, null, 2)}`);
      }
    }

  } catch (error) {
    console.log(`❌ 查詢失敗: ${error.message}`);
  }
}

// 主函數
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length > 0) {
    // 測試單一查詢
    await testSingleQuery(args.join(' '));
  } else {
    // 執行完整測試
    await testRAGWebIntegration();
  }
}

// 執行測試
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { testRAGWebIntegration, testSingleQuery }; 