# 醫師資料前處理與向量資料庫建立技術分析

## 概述

這個專案實現了一個完整的醫師資料前處理和向量資料庫建立系統，主要用於建立可搜尋的醫師資訊知識庫。系統整合了多項 NLP 技術和資料處理方法，提供了智能化的醫師資料管理解決方案。

## 系統架構

### 核心類別設計

```python
class DoctorDataProcessor:
    - 資料來源管理
    - 同義詞字典
    - 醫學術語模式匹配
    - 資料驗證與清理
    - 標準化處理
```

### 主要功能模組

1. **資料預處理模組**
2. **NLP 處理模組**
3. **向量化模組**
4. **資料庫操作模組**
5. **統計分析模組**

## 資料前處理技術

### 1. 資料清理 (Data Cleaning)

#### 文字標準化
- **去除冗餘空白**: 使用正則表達式 `r'\s+'` 統一空白字符
- **編號前綴清理**: 移除列表項目的編號格式如 "1)", "2)"
- **特殊標記處理**: 清除 HTML 標籤格式如 `<執照>`, `<學會>`
- **標點符號統一**: 中英文標點符號標準化

```python
def clean_text(self, text: str) -> str:
    # 移除多餘空白
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 移除編號前綴
    text = re.sub(r'^\d+\)', '', text)
    
    # 標準化標點符號
    text = text.replace('，', '，').replace('。', '。')
```

#### 日期格式標準化
- **民國年轉換**: 民國年份轉換為西元年份
- **日期格式統一**: 多種日期格式標準化為統一格式
- **時間範圍提取**: 從文本中提取服務年份範圍

### 2. 資料驗證 (Data Validation)

#### 完整性檢查
- **必要欄位驗證**: 確保 `name`, `department` 等核心欄位存在
- **資料品質檢查**: 檢查專長或職稱至少有一項
- **格式正確性**: 驗證資料格式符合預期結構

```python
def validate_doctor_data(self, doctor: Dict[str, Any]) -> bool:
    required_fields = ['name', 'department']
    for field in required_fields:
        if not doctor.get(field):
            return False
    return True
```

## NLP 處理技術

### 1. 同義詞標準化 (Synonym Standardization)

#### 醫學術語同義詞庫
- **職稱同義詞**: 主治醫師、主治、attending physician 等
- **專科同義詞**: 心臟血管內科、心臟內科、心血管內科等
- **醫院名稱標準化**: 不同表達方式的醫院名稱統一
- **學位認證同義詞**: 各種學位和認證的不同表達

```python
self.synonym_dict = {
    "心臟血管內科": ["心臟血管內科", "心臟內科", "心血管內科", "cardiology"],
    "主治醫師": ["主治醫師", "主治", "attending physician"],
    # ... 更多同義詞
}
```

### 2. 正則表達式模式匹配 (Pattern Matching)

#### 醫學資訊提取
- **年份提取**: 匹配西元年和民國年格式
- **學位識別**: 識別各種醫學學位
- **專科認證**: 提取專科醫師認證資訊
- **職位階級**: 識別醫院內職位層級

```python
self.medical_patterns = {
    'years': r'\b(19|20)\d{2}\b|民國\d{1,2}年',
    'medical_degrees': r'(學士|碩士|博士|Bachelor|Master|Doctor|PhD)',
    'medical_specialties': r'(內科|外科|婦產科|小兒科|精神科|...)',
    'positions': r'(主任|副主任|主治醫師|住院醫師|總醫師|...)'
}
```

### 3. 資訊提取與結構化 (Information Extraction)

#### 關鍵資訊萃取
- **醫學資訊解析**: 從非結構化文本中提取結構化醫學資訊
- **服務經歷分析**: 分析醫師的服務經歷和時間軸
- **主要醫院識別**: 識別醫師主要服務的醫院
- **經驗年份計算**: 計算醫師的服務年份範圍

```python
def extract_medical_info(self, text: str) -> Dict[str, List[str]]:
    info = {
        'years': [],
        'degrees': [],
        'specialties': [],
        'certifications': [],
        'positions': []
    }
    
    for key, pattern in self.medical_patterns.items():
        matches = re.findall(pattern, text)
        # 處理匹配結果
    
    return info
```

## 向量化技術

### 1. 語義向量模型

#### BGE-M3 多語言模型
- **模型選擇**: 使用 BAAI/bge-m3 作為向量化模型
- **多語言支援**: 支援中英文醫學術語
- **語義理解**: 能夠理解醫學專業術語的語義關係

```python
model = SentenceTransformer('BAAI/bge-m3')
embedding = model.encode(text)
```

### 2. 智能文本切割 (Smart Chunking)

#### 基於內容的切割策略
- **基本資訊片段**: 醫師基本資訊和主要專長
- **專長分組**: 長專長列表按語義分組
- **職位經歷**: 按時間軸分組職位經歷
- **認證資訊**: 專科認證和學歷分組

```python
def chunk_doctor_data_smart(doctor: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    
    # 1. 基本資訊片段
    basic_info = f"{doctor['name']} 醫師，專科：{doctor['department']}"
    chunks.append({'text': basic_info, 'chunk_type': 'basic_info'})
    
    # 2. 專長分組處理
    if len(doctor['specialty']) > 8:
        specialty_groups = [doctor['specialty'][i:i+8] for i in range(0, len(doctor['specialty']), 8)]
        # 為每組建立片段
    
    return chunks
```

## 資料庫技術

### 1. 向量資料庫 (ChromaDB)

#### 持久化儲存
- **向量索引**: 高效的向量相似度搜尋
- **元數據管理**: 豐富的醫師資訊元數據
- **多維查詢**: 支援多種查詢條件組合

```python
client = chromadb.PersistentClient(path=db_path)
collection = client.create_collection(collection_name)

collection.add(
    documents=[text],
    embeddings=[embedding.tolist()],
    metadatas=[metadata],
    ids=[unique_id]
)
```

### 2. 元數據設計

#### 結構化元數據
- **醫師基本資訊**: 姓名、科別、醫院
- **切割資訊**: 片段類型、分組索引
- **來源追蹤**: 資料來源、匯入時間
- **統計資訊**: 專長數量、職稱數量

## 搜尋最佳化技術

### 1. 多層次搜尋文本

#### 搜尋文本建立
- **關鍵詞提取**: 提取最重要的醫學關鍵詞
- **語義增強**: 加入同義詞和相關術語
- **結構化描述**: 按重要性組織搜尋文本

```python
def create_search_text(self, doctor: Dict[str, Any]) -> str:
    search_parts = []
    
    if doctor['name']:
        search_parts.append(f"醫師：{doctor['name']}")
    
    if doctor['specialty']:
        search_parts.append(f"專長：{' '.join(doctor['specialty'])}")
    
    return '。'.join(search_parts)
```

### 2. 相似度計算最佳化

#### 向量搜尋策略
- **語義相似度**: 使用 cosine similarity 計算
- **多候選結果**: 返回多個相關結果
- **閾值過濾**: 設定相似度閾值提高精確度

## 統計分析功能

### 1. 資料統計

#### 全面的統計報告
- **醫師數量統計**: 總數、科別分布
- **醫院分布**: 各醫院醫師數量
- **專長分析**: 專長領域分布和頻率
- **品質指標**: 平均專長數、職稱數等

```python
def generate_stats(processed_doctors: List[Dict]) -> Dict:
    stats = {
        'total_doctors': len(processed_doctors),
        'departments': Counter(d['department'] for d in processed_doctors),
        'hospitals': Counter(h for d in processed_doctors for h in d['main_hospitals']),
        'avg_specialties_per_doctor': sum(len(d['specialty']) for d in processed_doctors) / len(processed_doctors)
    }
    return stats
```

## 程式架構設計

### 1. 物件導向設計

#### 模組化架構
- **單一職責原則**: 每個類別和函數有明確的職責
- **依賴注入**: 通過參數傳遞依賴關係
- **錯誤處理**: 完善的異常處理機制

### 2. 配置管理

#### 靈活的配置系統
- **資料來源配置**: 可指定不同的資料來源
- **處理策略選擇**: 基本處理 vs 智能切割
- **輸出格式控制**: 自定義輸出格式和命名

### 3. 日誌系統

#### 完整的日誌記錄
- **處理進度追蹤**: 記錄每個處理步驟
- **錯誤日誌**: 詳細的錯誤信息和堆疊追蹤
- **統計日誌**: 處理結果統計和性能指標

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"已處理醫師 {idx+1}/{len(doctors)}: {processed_doctor['name']}")
```

## 技術特色

### 1. 中文醫學術語處理
- **專業術語庫**: 建立完整的醫學術語同義詞庫
- **中文分詞**: 針對中文醫學術語的特殊處理
- **語義理解**: 理解醫學專業術語的語義關係

### 2. 智能化資料處理
- **自適應切割**: 根據資料內容動態調整切割策略
- **品質驗證**: 多層次的資料品質檢查
- **錯誤恢復**: 遇到問題時的自動恢復機制

### 3. 可擴展性設計
- **模組化設計**: 易於添加新的處理模組
- **配置化系統**: 通過配置文件調整系統行為
- **API 化接口**: 提供標準化的 API 接口

## 應用場景

### 1. 醫師資訊查詢系統
- **智能搜尋**: 根據專長、醫院、職稱等多維度搜尋
- **相似醫師推薦**: 基於向量相似度的醫師推薦
- **專長匹配**: 根據病症匹配合適的醫師

### 2. 醫療知識管理
- **知識庫建立**: 建立結構化的醫療知識庫
- **資訊整合**: 整合多個來源的醫師資訊
- **資料分析**: 分析醫療資源分布和專長分布

### 3. 醫院管理系統
- **人力資源管理**: 醫師資源的統計和分析
- **專長分析**: 醫院專長分布和缺口分析
- **品質控制**: 醫師資料的品質監控

## 總結

這個系統整合了多項先進的 NLP 技術和資料處理方法，包括：

- **資料清理和標準化**：確保資料品質和一致性
- **語義理解**：使用先進的向量模型理解醫學術語
- **智能切割**：根據內容特性優化資料組織
- **向量搜尋**：提供高效的語義搜尋能力
- **統計分析**：提供全面的資料分析和報告

系統設計遵循軟體工程最佳實踐，具有良好的可維護性、可擴展性和錯誤處理能力，適合在實際的醫療資訊系統中部署使用。