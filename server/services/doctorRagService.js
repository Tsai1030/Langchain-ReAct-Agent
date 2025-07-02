const fs = require('fs');
const path = require('path');

// 醫師 RAG 服務 - 純 JavaScript 實現
class DoctorRAGService {
  constructor() {
    this.doctors = this.loadDoctors();
  }

  // 載入醫師資料
  loadDoctors() {
    try {
      const doctorsPath = path.join(__dirname, '../../doctors.json');
      const data = fs.readFileSync(doctorsPath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      console.error('❌ 載入醫師資料失敗:', error.message);
      return [];
    }
  }

  // 執行醫師檢索
  async searchDoctors(query) {
    console.log('🔍 執行醫師 RAG 檢索...');
    
    try {
      const relevantDoctors = this.findRelevantDoctors(query);
      
      console.log(`✅ 找到 ${relevantDoctors.length} 位相關醫師`);
      
      return {
        success: true,
        results: relevantDoctors,
        count: relevantDoctors.length,
        query: query
      };
    } catch (error) {
      console.error('❌ RAG 檢索失敗:', error.message);
      return {
        success: false,
        results: [],
        count: 0,
        error: error.message
      };
    }
  }

  // 根據查詢找到相關醫師
  findRelevantDoctors(query) {
    const lowerQuery = query.toLowerCase();
    const relevantDoctors = [];

    // 關鍵字權重配置
    const keywordWeights = {
      // 醫師姓名（最高權重）
      name: 10,
      // 科別
      department: 8,
      // 專長
      specialty: 6,
      // 職稱
      title: 4,
      // 經驗
      experience: 3,
      // 教育背景
      education: 2,
      // 證照
      certifications: 2
    };

    this.doctors.forEach(doctor => {
      let score = 0;
      let matchedFields = [];

      // 檢查醫師姓名
      if (lowerQuery.includes(doctor.name.toLowerCase())) {
        score += keywordWeights.name;
        matchedFields.push('姓名');
      }

      // 檢查科別
      if (doctor.department && lowerQuery.includes(doctor.department.toLowerCase())) {
        score += keywordWeights.department;
        matchedFields.push('科別');
      }

      // 檢查專長
      if (doctor.specialty && Array.isArray(doctor.specialty)) {
        doctor.specialty.forEach(spec => {
          if (lowerQuery.includes(spec.toLowerCase())) {
            score += keywordWeights.specialty;
            matchedFields.push('專長');
          }
        });
      }

      // 檢查職稱
      if (doctor.title && Array.isArray(doctor.title)) {
        doctor.title.forEach(title => {
          if (lowerQuery.includes(title.toLowerCase())) {
            score += keywordWeights.title;
            matchedFields.push('職稱');
          }
        });
      }

      // 檢查經驗
      if (doctor.experience && Array.isArray(doctor.experience)) {
        doctor.experience.forEach(exp => {
          if (lowerQuery.includes(exp.toLowerCase())) {
            score += keywordWeights.experience;
            matchedFields.push('經驗');
          }
        });
      }

      // 檢查教育背景
      if (doctor.education && Array.isArray(doctor.education)) {
        doctor.education.forEach(edu => {
          if (lowerQuery.includes(edu.toLowerCase())) {
            score += keywordWeights.education;
            matchedFields.push('教育');
          }
        });
      }

      // 檢查證照
      if (doctor.certifications && Array.isArray(doctor.certifications)) {
        doctor.certifications.forEach(cert => {
          if (lowerQuery.includes(cert.toLowerCase())) {
            score += keywordWeights.certifications;
            matchedFields.push('證照');
          }
        });
      }

      // 如果分數大於 0，加入結果
      if (score > 0) {
        relevantDoctors.push({
          ...doctor,
          relevanceScore: score,
          matchedFields: [...new Set(matchedFields)], // 去重
          relevance: Math.min(score / 20, 1) // 正規化到 0-1
        });
      }
    });

    // 按相關度排序
    relevantDoctors.sort((a, b) => b.relevanceScore - a.relevanceScore);

    // 返回前 3 位最相關的醫師
    return relevantDoctors.slice(0, 3);
  }

  // 格式化醫師資訊
  formatDoctorInfo(doctor) {
    const info = [];
    
    info.push(`👨‍⚕️ **${doctor.name}** - ${doctor.department}`);
    
    if (doctor.title && doctor.title.length > 0) {
      info.push(`📋 **職稱**: ${doctor.title.join(', ')}`);
    }
    
    if (doctor.specialty && doctor.specialty.length > 0) {
      info.push(`🎯 **專長**: ${doctor.specialty.join(', ')}`);
    }
    
    if (doctor.experience && doctor.experience.length > 0) {
      info.push(`💼 **經歷**: ${doctor.experience.slice(0, 3).join(', ')}`);
    }
    
    if (doctor.education && doctor.education.length > 0) {
      info.push(`🎓 **學歷**: ${doctor.education.join(', ')}`);
    }
    
    return info.join('\n');
  }

  // 建立整合的醫師資訊摘要
  createDoctorSummary(ragResults) {
    if (!ragResults.success || ragResults.count === 0) {
      return null;
    }

    const summary = {
      totalDoctors: ragResults.count,
      doctors: ragResults.results.map(doctor => ({
        name: doctor.name,
        department: doctor.department,
        specialty: doctor.specialty ? doctor.specialty.join(', ') : '',
        title: doctor.title ? doctor.title[0] : '',
        relevance: doctor.relevance
      })),
      formattedInfo: ragResults.results.map(doctor => this.formatDoctorInfo(doctor))
    };

    return summary;
  }
}

module.exports = DoctorRAGService; 