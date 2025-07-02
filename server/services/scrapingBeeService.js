const axios = require('axios');
const cheerio = require('cheerio');

// ScrapingBee 動態網頁搜尋服務
class ScrapingBeeService {
  constructor() {
    this.apiKey = process.env.SCRAPING_BEE_KEY;
    this.baseURL = 'https://app.scrapingbee.com/api/v1';
    
    if (!this.apiKey) {
      console.warn('⚠️ 警告：SCRAPING_BEE_KEY 環境變數未設定');
      console.warn('動態網頁搜尋功能將被停用');
    }

    // 科別名稱與 virtualDept 代碼對照表
    this.deptCodeMap = {
      '內科部': '0100',
      '外科部': '0200',
      '婦產部': '0300',
      '小兒部': '0400',
      '眼科部': '0500',
      '耳鼻喉部': '0600',
      '骨科部': '0700',
      '泌尿部': '0800',
      '皮膚部': '0900',
      '神經部': '1000',
      '精神醫學部': '1100',
      '放射腫瘤部': '1200',
      '牙科部': '1300',
      '家庭醫學科': '1500',
      '急診部': '1600',
      '疼痛科': '1700',
      '復健部': '1800',
      '健康管理中心': '1900',
      '職業及環境醫學科': '2100',
      '癌症中心': '2500',
      '社區醫學部': '2600',
      '中醫部': '2800',
      '特別門診': '2900',
      '影像醫學部': '3300',
      '麻醉部': '3400',
      '藥學部': '4600',
      '特殊血液病防治中心': '9909',
      '一般門診': '9925',
      '臨床試驗門診': '9926',
      '檢驗醫學基因診斷門診': '9927',
      '老人健康照護計畫門診': '9928',
      '多重抗藥結核特診': '9940',
      '共病族群潛伏結核感染治療特診': '9941',
      '外籍服藥支持計畫診': '9943',
    };
  }

  // 檢查是否為即時資訊查詢
  isRealTimeQuery(query) {
    const realTimeKeywords = [
      '看到幾號', '叫號', '現在', '目前', '即時', 
      '等待時間', '排隊', '進度', '現在看到', '當前'
    ];
    
    return realTimeKeywords.some(keyword => query.includes(keyword));
  }

  // 提取醫院名稱
  extractHospitalName(query) {
    const hospitals = {
      '高醫': '高醫',
      '高雄醫學大學': '高醫',
      '高雄醫大': '高醫',
      '台大': '台大',
      '台大醫院': '台大',
      '長庚': '長庚',
      '榮總': '榮總',
      '三總': '三總'
    };

    for (const [key, value] of Object.entries(hospitals)) {
      if (query.includes(key)) {
        return value;
      }
    }
    return null;
  }

  // 解析查詢句，精準拆解科別與醫師
  parseQueryInfo(query) {
    // 動態組合所有科別名稱
    const deptNames = Object.keys(this.deptCodeMap).map(name => name.replace(/部|科|診/g, '.?')).join('|');
    // 例如：(內科.?|外科.?|婦產科.?|...)
    const deptReg = new RegExp(`(${deptNames}[1１2２3３]?診)`);
    const deptMatch = query.match(deptReg);

    // 醫師名稱
    let doctor = '';
    const doctorReg = /診[\s]*([王李張陳林吳楊黃蔡鄭梁葉郭余][\u4e00-\u9fa5]{1,3}?)(?=目前|現在|的|\s|$)/;
    const doctorMatch = query.match(doctorReg);
    if (doctorMatch) {
      doctor = doctorMatch[1];
    } else {
      // fallback: 找第一個常見姓氏開頭的2~3字
      const fallbackMatch = query.match(/([王李張陳林吳楊黃蔡鄭梁葉郭余][\u4e00-\u9fa5]{1,2})/);
      doctor = fallbackMatch ? fallbackMatch[1] : '';
    }
    return {
      dept: deptMatch ? deptMatch[0] : '',
      doctor
    };
  }

  // 取得診間進度，支援全形/半形數字與空白
  extractClinicProgress(html, { dept, doctor }) {
    const $ = cheerio.load(html);
    let result = null;
    const normalize = str => (str || '').replace(/[　\s]/g, '').replace('１', '1').replace('２', '2').replace('３', '3');

    $('.c_table').each((i, el) => {
      const deptName = normalize($(el).find('.SeqDept').text());
      const doctorName = normalize($(el).find('.DocName').text());
      const currentSeq = $(el).find('.CurrentSeq').text().trim();
      console.log('診間:', deptName, '醫師:', doctorName, '號碼:', currentSeq, '查詢科別:', normalize(dept), '查詢醫師:', normalize(doctor));

      if (
        (!dept || deptName.includes(normalize(dept)))
        && (!doctor || doctorName.includes(normalize(doctor)))
      ) {
        result = {
          dept: deptName,
          doctor: doctorName,
          currentSeq
        };
        return false; // break
      }
    });

    return result || null;
  }

  // 獲取醫院即時叫號資訊
  async getHospitalQueueInfo(hospital, query) {
    // 只支援高醫
    if (hospital !== '高醫') {
      throw new Error('目前僅支援高醫即時掛號查詢');
    }
    const url = 'https://www.kmuh.org.tw/Web/WebRegistration/OPDSeq/ProcessMain?lang=tw';
    if (!this.apiKey) {
      throw new Error('ScrapingBee API key 未設定');
    }
    try {
      const response = await axios.get(this.baseURL, {
        params: {
          api_key: this.apiKey,
          url: url,
          render_js: true,
          wait: 8000,
          country_code: 'tw',
          premium_proxy: true
        },
        timeout: 30000
      });
      const html = response.data;
      const queryInfo = this.parseQueryInfo(query);
      const clinicProgress = this.extractClinicProgress(html, queryInfo);
      if (clinicProgress) {
        console.log('回傳 clinicProgress:', clinicProgress);
        return {
          hospital,
          department: clinicProgress.dept,
          doctor: clinicProgress.doctor,
          currentNumber: clinicProgress.currentSeq,
          success: true,
          timestamp: new Date().toISOString()
        };
      }
    } catch (error) {
      console.log(`❌ 嘗試 ${url} 失敗: ${error.message}`);
    }
    // 若都沒抓到
    console.log('⚠️ 無法找到叫號資訊，使用模擬資料');
    return this.getMockData(hospital);
  }

  // 分析 HTML 內容尋找叫號資訊
  analyzeHTMLForQueueInfo(html, hospital) {
    try {
      console.log('🔍 分析 HTML 內容尋找叫號資訊...');
      
      // 將 HTML 轉換為小寫以便搜尋
      const lowerHtml = html.toLowerCase();
      
      // 搜尋可能的叫號相關文字
      const queuePatterns = [
        /(\d+)\s*號/g,           // 數字+號
        /current.*?(\d+)/gi,     // current + 數字
        /now.*?(\d+)/gi,         // now + 數字
        /(\d+).*?current/gi,     // 數字 + current
        /(\d+).*?now/gi,         // 數字 + now
        /叫號.*?(\d+)/g,         // 叫號 + 數字
        /(\d+).*?叫號/g,         // 數字 + 叫號
        /等待.*?(\d+)/g,         // 等待 + 數字
        /(\d+).*?等待/g          // 數字 + 等待
      ];

      const foundNumbers = [];
      
      // 搜尋所有可能的數字
      queuePatterns.forEach(pattern => {
        let match;
        while ((match = pattern.exec(lowerHtml)) !== null) {
          foundNumbers.push(parseInt(match[1]));
        }
      });

      // 搜尋科別資訊
      const deptPatterns = [
        /心臟內科/,
        /心臟科/,
        /內科/,
        /外科/,
        /急診/,
        /門診/
      ];

      let foundDepartment = '';
      deptPatterns.forEach(pattern => {
        if (pattern.test(html)) {
          foundDepartment = pattern.source;
        }
      });

      console.log(`📊 找到的數字: ${foundNumbers.join(', ')}`);
      console.log(`📊 找到的科別: ${foundDepartment}`);

      if (foundNumbers.length > 0) {
        return {
          found: true,
          currentNumber: foundNumbers[0].toString(),
          nextNumber: foundNumbers.length > 1 ? foundNumbers[1].toString() : '無法取得',
          department: foundDepartment || '無法取得'
        };
      }

      return { found: false };
      
    } catch (error) {
      console.error('❌ HTML 分析失敗:', error);
      return { found: false };
    }
  }

  // 模擬資料 (當無法獲取真實資料時)
  getMockData(hospital) {
    const mockData = {
      hospital: hospital,
      timestamp: new Date().toISOString(),
      currentNumber: '無法取得即時資料',
      nextNumber: '無法取得即時資料',
      department: '無法取得',
      source: 'mock-data',
      success: false,
      message: '建議直接聯繫醫院或查看醫院官方叫號系統'
    };

    console.log(`⚠️ 使用模擬資料: ${hospital}`);
    return mockData;
  }

  // 混合搜尋：結合即時 API 和 Google 搜尋
  async hybridSearch(query, googleResults) {
    // 檢查是否為即時查詢
    if (!this.isRealTimeQuery(query)) {
      return {
        type: 'static',
        data: googleResults,
        message: '使用 Google 搜尋結果'
      };
    }

    // 提取醫院名稱
    const hospital = this.extractHospitalName(query);
    if (!hospital) {
      return {
        type: 'static',
        data: googleResults,
        message: '無法識別醫院，使用 Google 搜尋結果'
      };
    }

    try {
      // 嘗試獲取即時資訊
      const realTimeData = await this.getHospitalQueueInfo(hospital, query);
      
      if (realTimeData.success && realTimeData.currentNumber !== '無法取得即時資料') {
        return {
          type: 'real-time',
          data: realTimeData,
          message: '成功獲取即時資訊'
        };
      } else {
        return {
          type: 'fallback',
          data: googleResults,
          message: realTimeData.message || '無法獲取即時資訊，使用 Google 搜尋結果'
        };
      }
    } catch (error) {
      console.log('⚠️ 即時資訊獲取失敗，使用 Google 搜尋結果');
      return {
        type: 'fallback',
        data: googleResults,
        message: `即時資訊獲取失敗: ${error.message}，使用 Google 搜尋結果`
      };
    }
  }

  // 取得午別（預設下午診）
  parseNoon(query) {
    if (query.includes('上午')) return 'AM';
    if (query.includes('下午')) return 'PM';
    if (query.includes('夜診')) return 'Night';
    return 'PM';
  }
}

module.exports = ScrapingBeeService;