const { spawn } = require('child_process');
const path = require('path');

// 向量 RAG 服務 - 使用 Python 向量檢索
class VectorRAGService {
  constructor() {
    this.scriptPath = path.join(__dirname, '../../test_vector_rag.py');
  }

  // 智能檢測 Python 路徑
  findPythonPath() {
    const possiblePaths = [
      'python',
      'python3',
      'C:\\Users\\user\\anaconda3\\envs\\doctor-rag\\python.exe',
      'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python39\\python.exe',
      'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python310\\python.exe',
      'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
    ];

    // 直接返回第一個可能的路徑，讓實際執行時處理錯誤
    // 這樣可以避免在檢測階段就失敗
    for (const pythonPath of possiblePaths) {
      console.log(`🔍 嘗試 Python 路徑: ${pythonPath}`);
      return pythonPath;
    }

    console.log('⚠️ 無法找到 Python，使用預設路徑: python');
    return 'python';
  }

  // 執行向量檢索
  async searchDoctors(query) {
    console.log('🔍 執行向量 RAG 檢索...');
    
    // 每次執行時都檢測 Python 路徑
    const pythonPath = this.findPythonPath();
    
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn(pythonPath, [this.scriptPath, query], {
        cwd: path.join(__dirname, '../..'),
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            // 解析 Python 輸出，提取醫師資訊
            const doctors = this.parsePythonOutput(output, query);
            console.log(`✅ 向量 RAG 檢索完成，找到 ${doctors.length} 位相關醫師`);
            
            resolve({
              success: true,
              results: doctors,
              count: doctors.length,
              query: query,
              method: 'vector'
            });
          } catch (error) {
            console.log(`❌ 解析向量檢索結果失敗: ${error.message}`);
            resolve({ success: false, results: [], count: 0, method: 'vector' });
          }
        } else {
          console.log(`❌ 向量 RAG 檢索執行失敗 (code: ${code})`);
          console.log(`錯誤輸出: ${errorOutput}`);
          resolve({ success: false, results: [], count: 0, method: 'vector' });
        }
      });

      pythonProcess.on('error', (error) => {
        console.log(`❌ 向量 RAG 檢索啟動失敗: ${error.message}`);
        resolve({ success: false, results: [], count: 0, method: 'vector' });
      });
    });
  }

  // 解析 Python 輸出，提取醫師資訊
  parsePythonOutput(output, query) {
    const doctors = [];
    const lines = output.split('\n');
    let currentDoctor = null;
    for (const line of lines) {
      // 解析相似度行
      if (line.match(/相似度:/)) {
        const similarityMatch = line.match(/相似度: ([\d.\-]+)/);
        const similarity = similarityMatch ? parseFloat(similarityMatch[1]) : 0;
        currentDoctor = { relevance: similarity };
      }
      // 解析完整內容行
      if (line.match(/^\s*完整內容:/) && currentDoctor) {
        const content = line.replace(/^\s*完整內容:/, '').trim();
        const doctorInfo = this.extractDoctorInfo(content);
        Object.assign(currentDoctor, doctorInfo);
        if (currentDoctor.name) {
          doctors.push(currentDoctor);
        }
        currentDoctor = null;
      }
    }
    // 按相似度排序
    doctors.sort((a, b) => b.relevance - a.relevance);
    return doctors.slice(0, 3);
  }

  // 從文本中提取醫師資訊
  extractDoctorInfo(content) {
    const parts = content.split('，');
    if (parts.length < 2) return {};
    
    const name = parts[0];
    const department = parts[1];
    
    // 提取專長（通常在科別之後）
    const specialty = [];
    for (let i = 2; i < parts.length; i++) {
      const part = parts[i];
      if (part.includes('醫師') || part.includes('主任') || part.includes('教授')) {
        break; // 遇到職稱就停止
      }
      if (part && !part.includes('高雄醫學大學') && !part.includes('附設')) {
        specialty.push(part);
      }
    }
    
    // 提取職稱
    const title = [];
    for (const part of parts) {
      if (part.includes('醫師') || part.includes('主任') || part.includes('教授')) {
        title.push(part);
      }
    }
    
    return {
      name,
      department,
      specialty: specialty.slice(0, 5), // 限制專長數量
      title: title.slice(0, 3), // 限制職稱數量
      relevance: 0
    };
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
    
    info.push(`🔍 **相關度**: ${doctor.relevance.toFixed(3)}`);
    
    return info.join('\n');
  }

  // 建立整合的醫師資訊摘要
  createDoctorSummary(ragResults) {
    if (!ragResults.success || ragResults.count === 0) {
      return null;
    }

    const summary = {
      totalDoctors: ragResults.count,
      method: 'vector',
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

module.exports = VectorRAGService; 