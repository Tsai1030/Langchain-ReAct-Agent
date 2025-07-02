const fs = require('fs');
const path = require('path');

// 修復 OpenAI API timeout 參數問題
function fixTimeoutIssue() {
  const filePath = path.join(__dirname, 'server', 'services', 'queryService.js');
  
  try {
    // 讀取檔案
    let content = fs.readFileSync(filePath, 'utf8');
    
    // 移除 timeout 參數
    content = content.replace(/,\s*timeout:\s*30000\s*\/\/\s*30\s*秒超時/g, '');
    
    // 寫回檔案
    fs.writeFileSync(filePath, content, 'utf8');
    
    console.log('✅ 已修復 OpenAI API timeout 參數問題');
    console.log('📝 移除了不支援的 timeout 參數');
    
  } catch (error) {
    console.error('❌ 修復失敗:', error.message);
  }
}

// 執行修復
fixTimeoutIssue(); 