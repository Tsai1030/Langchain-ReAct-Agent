const axios = require('axios');

// 測試配置
const API_BASE_URL = 'http://localhost:3001';
const TEST_QUERY = '今天高醫心臟內科看到幾號？';

async function testQuery() {
  try {
    console.log('🧪 開始測試醫療資訊查詢系統...');
    console.log(`📝 測試查詢: "${TEST_QUERY}"`);
    console.log('⏳ 正在發送請求...\n');

    const response = await axios.post(`${API_BASE_URL}/api/query`, {
      query: TEST_QUERY
    }, {
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    console.log('✅ 查詢成功！');
    console.log('📊 回應內容:');
    console.log('='.repeat(50));
    console.log(`原始查詢: ${response.data.query}`);
    console.log(`AI 回應: ${response.data.response}`);
    console.log(`搜尋結果數量: ${response.data.searchResults.totalResults}`);
    console.log(`時間戳記: ${response.data.timestamp}`);
    console.log('='.repeat(50));

    if (response.data.searchResults.organic.length > 0) {
      console.log('\n🔍 搜尋結果摘要:');
      response.data.searchResults.organic.forEach((result, index) => {
        console.log(`${index + 1}. ${result.title}`);
        console.log(`   網址: ${result.link}`);
        console.log(`   摘要: ${result.snippet.substring(0, 100)}...`);
        console.log('');
      });
    }

  } catch (error) {
    console.error('❌ 測試失敗:');
    if (error.response) {
      console.error(`狀態碼: ${error.response.status}`);
      console.error(`錯誤訊息: ${error.response.data.error || error.response.data.message}`);
    } else if (error.request) {
      console.error('網路連線錯誤，請確認後端伺服器是否正在運行');
    } else {
      console.error(`請求錯誤: ${error.message}`);
    }
  }
}

async function testHealth() {
  try {
    console.log('🏥 測試健康檢查端點...');
    const response = await axios.get(`${API_BASE_URL}/health`);
    console.log('✅ 健康檢查通過');
    console.log(`狀態: ${response.data.status}`);
    console.log(`時間: ${response.data.timestamp}`);
    console.log('');
  } catch (error) {
    console.error('❌ 健康檢查失敗');
    console.error('請確認後端伺服器是否正在運行在 http://localhost:3001');
  }
}

// 執行測試
async function runTests() {
  console.log('🚀 醫療資訊查詢系統 - API 測試');
  console.log('='.repeat(50));
  
  await testHealth();
  await testQuery();
  
  console.log('🎉 測試完成！');
}

// 如果直接執行此檔案
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { testQuery, testHealth }; 