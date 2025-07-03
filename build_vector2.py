import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from collections import Counter

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoctorDataProcessor:
    def __init__(self, data_source: str = "doctors3.json"):
        self.data_source = data_source
        self.import_date = datetime.now().strftime("%Y-%m-%d")
        self.import_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 擴展的同義詞字典
        self.synonym_dict = {
            # 職稱同義詞
            "主治醫師": ["主治醫師", "主治", "attending physician", "主治醫生"],
            "住院醫師": ["住院醫師", "住院", "resident", "住院醫生"],
            "總醫師": ["總醫師", "總住院醫師", "總", "chief resident"],
            "主任": ["主任", "director", "chief", "科主任", "部主任"],
            "副主任": ["副主任", "associate director", "副科主任"],
            "教授": ["教授", "professor", "prof"],
            "副教授": ["副教授", "associate professor"],
            "助理教授": ["助理教授", "assistant professor"],
            "講師": ["講師", "lecturer"],
            "醫務秘書": ["醫務秘書", "醫務祕書"],
            
            # 專科同義詞
            "心臟血管內科": ["心臟血管內科", "心臟內科", "心血管內科", "cardiology", "心內科", "心臟科"],
            "循環學": ["循環學", "心血管學", "cardiovascular", "循環系統"],
            "高血壓": ["高血壓", "hypertension", "HTN", "血壓高"],
            "心衰竭": ["心衰竭", "心臟衰竭", "heart failure", "HF", "心功能不全"],
            "心肌梗塞": ["心肌梗塞", "心梗", "myocardial infarction", "MI", "心肌梗死"],
            "心絞痛": ["心絞痛", "angina", "胸痛"],
            "心律不整": ["心律不整", "心律不齊", "arrhythmia", "心律失常"],
            "心導管": ["心導管", "cardiac catheterization", "心導管檢查"],
            "血管支架": ["血管支架", "stent", "支架置入"],
            "冠狀動脈": ["冠狀動脈", "coronary artery", "冠心病"],
            "心臟超音波": ["心臟超音波", "心臟超音波檢查", "echocardiography", "心臟超音波檢查"],
            "介入性治療": ["介入性治療", "介入治療", "interventional therapy", "心導管介入治療"],
            "重症醫學": ["重症醫學", "重症照護", "intensive care", "加護病房"],
            "肺動脈高壓": ["肺動脈高壓", "pulmonary hypertension", "肺高壓"],
            "心肌病變": ["心肌病變", "cardiomyopathy", "心肌病"],
            "瓣膜性心臟病": ["瓣膜性心臟病", "valvular heart disease", "心臟瓣膜疾病"],
            "高血脂": ["高血脂", "hyperlipidemia", "血脂異常"],
            "周邊血管": ["周邊血管", "peripheral vascular", "周邊動脈"],
            "深層靜脈栓塞": ["深層靜脈栓塞", "deep vein thrombosis", "DVT"],
            "心房中膈缺損": ["心房中膈缺損", "atrial septal defect", "ASD"],
            "心室中膈缺損": ["心室中膈缺損", "ventricular septal defect", "VSD"],
            "開放性動脈導管": ["開放性動脈導管", "patent ductus arteriosus", "PDA"],
            "主動脈瓣膜": ["主動脈瓣膜", "aortic valve", "主動脈瓣"],
            "二尖瓣": ["二尖瓣", "mitral valve", "僧帽瓣"],
            
            # 醫院名稱標準化
            "高雄醫學大學附設中和紀念醫院": [
                "高雄醫學大學附設中和紀念醫院", "高醫附院", "中和醫院", "中和紀念醫院",
                "高雄醫學大學附設醫院", "高醫中和醫院", "高雄醫學大學附設中和紀念醫院"
            ],
            "高雄醫學大學附設高醫岡山醫院": [
                "高雄醫學大學附設高醫岡山醫院", "高醫岡山醫院", "岡山醫院"
            ],
            "高雄市立小港醫院": ["高雄市立小港醫院", "小港醫院"],
            "高雄市立大同醫院": ["高雄市立大同醫院", "大同醫院"],
            
            # 學位同義詞
            "學士": ["學士", "bachelor", "學士學位", "醫學士"],
            "碩士": ["碩士", "master", "碩士學位"],
            "博士": ["博士", "doctor", "博士學位", "PhD"],
            
            # 專科認證同義詞
            "專科醫師": ["專科醫師", "專科", "specialist"],
            "指導醫師": ["指導醫師", "指導", "supervisor"],
            "會員": ["會員", "member"],
            "會士": ["會士", "fellow"],
            "證書": ["證書", "certificate", "執照"]
        }
        
        # 醫學專業術語正則表達式
        self.medical_patterns = {
            'years': r'\b(19|20)\d{2}\b|民國\d{1,2}年',
            'medical_degrees': r'(學士|碩士|博士|Bachelor|Master|Doctor|PhD|醫學士)',
            'medical_specialties': r'(內科|外科|婦產科|小兒科|精神科|皮膚科|眼科|耳鼻喉科|骨科|泌尿科|放射科|麻醉科|病理科|復健科|家醫科|心臟科|心臟血管內科)',
            'certifications': r'(專科醫師|指導醫師|會員|會士|證書|執照)',
            'positions': r'(主任|副主任|主治醫師|住院醫師|總醫師|教授|副教授|助理教授|講師)'
        }
    
    def clean_text(self, text: str) -> str:
        """增強的文字清理功能"""
        if not text:
            return ""
        
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除編號前綴 (如 "1)", "2)" 等)
        text = re.sub(r'^\d+\)', '', text)
        
        # 移除特殊標記 (如 "<執照>", "<學會>" 等)
        text = re.sub(r'^<[^>]+>', '', text)
        
        # 標準化標點符號
        text = text.replace('，', '，').replace('。', '。')
        text = text.replace('(', '（').replace(')', '）')
        text = text.replace('[', '［').replace(']', '］')
        
        # 標準化英文數字格式
        text = re.sub(r'(\d+)/(\d+)', r'\1年\2月', text)
        text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1年\2月\3日', text)
        
        # 移除特殊字符但保留醫學相關符號
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。（）［］\-/：、]', '', text)
        
        return text
    
    def validate_doctor_data(self, doctor: Dict[str, Any]) -> bool:
        """驗證醫師資料完整性"""
        required_fields = ['name', 'department']
        for field in required_fields:
            if not doctor.get(field):
                logger.warning(f"缺少必要欄位: {field}")
                return False
        
        # 檢查至少有一個專長或職稱
        if not doctor.get('specialty') and not doctor.get('title'):
            logger.warning(f"醫師 {doctor.get('name', 'Unknown')} 缺少專長或職稱資訊")
            return False
        
        return True
    
    def extract_medical_info(self, text: str) -> Dict[str, List[str]]:
        """提取醫學相關資訊"""
        info = {
            'years': [],
            'degrees': [],
            'specialties': [],
            'certifications': [],
            'positions': []
        }
        
        for key, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text)
            if key == 'years':
                info['years'] = matches
            elif key == 'medical_degrees':
                info['degrees'] = matches
            elif key == 'medical_specialties':
                info['specialties'] = matches
            elif key == 'certifications':
                info['certifications'] = matches
            elif key == 'positions':
                info['positions'] = matches
        
        return info
    
    def standardize_term(self, term: str) -> str:
        """標準化術語"""
        term = self.clean_text(term)
        
        # 檢查同義詞字典
        for standard_term, variations in self.synonym_dict.items():
            if any(var in term for var in variations):
                return standard_term
        
        return term
    
    def process_list_field(self, field_data: List[str]) -> List[str]:
        """增強的列表欄位處理"""
        if not field_data:
            return []
        
        processed = []
        for item in field_data:
            cleaned_item = self.clean_text(item)
            if cleaned_item:
                # 標準化術語
                standardized_item = self.standardize_term(cleaned_item)
                processed.append(standardized_item)
        
        # 去除重複項目但保持順序
        seen = set()
        unique_processed = []
        for item in processed:
            if item not in seen:
                seen.add(item)
                unique_processed.append(item)
        
        return unique_processed
    
    def extract_key_info(self, doctor: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """提取關鍵資訊"""
        # 驗證資料完整性
        if not self.validate_doctor_data(doctor):
            logger.warning(f"醫師資料驗證失敗，跳過處理: {doctor.get('name', 'Unknown')}")
            return {}
        
        processed_doctor = {}
        
        # 基本資訊
        processed_doctor['name'] = self.clean_text(doctor.get('name', ''))
        processed_doctor['department'] = self.clean_text(doctor.get('department', ''))
        
        # 處理列表欄位
        processed_doctor['specialty'] = self.process_list_field(doctor.get('specialty', []))
        processed_doctor['title'] = self.process_list_field(doctor.get('title', []))
        processed_doctor['experience'] = self.process_list_field(doctor.get('experience', []))
        processed_doctor['education'] = self.process_list_field(doctor.get('education', []))
        processed_doctor['certifications'] = self.process_list_field(doctor.get('certifications', []))
        
        # 提取醫學相關資訊
        all_text = ' '.join([
            processed_doctor['name'],
            processed_doctor['department'],
            ' '.join(processed_doctor['specialty']),
            ' '.join(processed_doctor['title']),
            ' '.join(processed_doctor['experience']),
            ' '.join(processed_doctor['education']),
            ' '.join(processed_doctor['certifications'])
        ])
        
        medical_info = self.extract_medical_info(all_text)
        processed_doctor.update(medical_info)
        
        # 提取主要醫院
        main_hospitals = set()
        for item in processed_doctor.get('title', []) + processed_doctor.get('experience', []):
            standardized = self.standardize_term(item)
            if any(hospital in standardized for hospital in self.synonym_dict.keys() if '醫院' in hospital):
                for hospital in self.synonym_dict.keys():
                    if '醫院' in hospital and hospital in standardized:
                        main_hospitals.add(hospital)
        
        processed_doctor['main_hospitals'] = list(main_hospitals)
        
        # 計算經驗年份範圍
        if processed_doctor['years']:
            years = []
            for year in processed_doctor['years']:
                # 處理民國年份
                if '民國' in year:
                    try:
                        ming_year = int(re.search(r'民國(\d{1,2})年', year).group(1))
                        western_year = ming_year + 1911
                        years.append(western_year)
                    except:
                        continue
                else:
                    try:
                        years.append(int(year))
                    except:
                        continue
            
            if years:
                processed_doctor['experience_years'] = {
                    'min_year': min(years),
                    'max_year': max(years),
                    'span': max(years) - min(years) if len(years) > 1 else 0
                }
            else:
                processed_doctor['experience_years'] = {}
        else:
            processed_doctor['experience_years'] = {}
        
        # 新增資料來源和日期資訊
        processed_doctor['data_source'] = self.data_source
        processed_doctor['import_date'] = self.import_date
        processed_doctor['import_timestamp'] = self.import_timestamp
        processed_doctor['record_id'] = f"{self.data_source}_{index}"
        
        return processed_doctor
    
    def create_search_text(self, doctor: Dict[str, Any]) -> str:
        """建立搜尋用文字 - 針對中文醫學術語優化"""
        search_parts = []
        
        # 基本資訊
        if doctor['name']:
            search_parts.append(f"醫師：{doctor['name']}")
        
        if doctor['department']:
            search_parts.append(f"科別：{doctor['department']}")
        
        # 專長（使用標準化術語）
        if doctor['specialty']:
            specialty_text = ' '.join(doctor['specialty'])
            search_parts.append(f"專長：{specialty_text}")
        
        # 重要職稱
        if doctor['title']:
            # 篩選重要職稱
            important_titles = []
            for title in doctor['title']:
                if any(keyword in title for keyword in ['主任', '教授', '副教授', '講師', '主治醫師']):
                    important_titles.append(title)
            
            if important_titles:
                search_parts.append(f"職稱：{' '.join(important_titles[:3])}")
        
        # 主要醫院
        if doctor['main_hospitals']:
            search_parts.append(f"醫院：{' '.join(doctor['main_hospitals'])}")
        
        # 重要專科認證
        if doctor['certifications']:
            cert_keywords = ['專科醫師', '指導醫師', '會士']
            important_certs = []
            for cert in doctor['certifications']:
                if any(keyword in cert for keyword in cert_keywords):
                    important_certs.append(cert)
            
            if important_certs:
                search_parts.append(f"認證：{' '.join(important_certs[:3])}")
        
        # 經驗年份
        if doctor.get('experience_years'):
            exp_years = doctor['experience_years']
            if exp_years.get('min_year') and exp_years.get('max_year'):
                search_parts.append(f"服務期間：{exp_years['min_year']}-{exp_years['max_year']}")
        
        return '。'.join(search_parts)
    
    def create_detailed_text(self, doctor: Dict[str, Any]) -> str:
        """建立詳細描述文字"""
        detailed_parts = []
        
        # 基本資訊
        detailed_parts.append(f"{doctor['name']} 醫師")
        detailed_parts.append(f"專科：{doctor['department']}")
        
        # 專長
        if doctor['specialty']:
            detailed_parts.append(f"專長領域：{' '.join(doctor['specialty'])}")
        
        # 現任職位
        if doctor['title']:
            detailed_parts.append(f"現任職位：{' '.join(doctor['title'])}")
        
        # 學歷
        if doctor['education']:
            detailed_parts.append(f"學歷：{' '.join(doctor['education'])}")
        
        # 專科認證
        if doctor['certifications']:
            detailed_parts.append(f"專科認證：{' '.join(doctor['certifications'])}")
        
        # 主要經歷
        if doctor['experience']:
            detailed_parts.append(f"主要經歷：{' '.join(doctor['experience'][:5])}")
        
        # 服務醫院
        if doctor['main_hospitals']:
            detailed_parts.append(f"服務醫院：{' '.join(doctor['main_hospitals'])}")
        
        return '。'.join(detailed_parts)

def chunk_doctor_data_smart(doctor: Dict[str, Any], processor: DoctorDataProcessor) -> List[Dict[str, Any]]:
    """
    智能切割醫師資料
    根據資料內容和長度採用不同的切割策略
    """
    chunks = []
    
    # 1. 基本資訊片段（始終創建）
    basic_info = f"{doctor['name']} 醫師，專科：{doctor['department']}"
    if doctor['specialty']:
        basic_info += f"，專長：{' '.join(doctor['specialty'][:3])}"  # 只取前3個專長
    
    chunks.append({
        'text': basic_info,
        'chunk_type': 'basic_info',
        'doctor_data': doctor
    })
    
    # 2. 專長片段（如果專長很多）
    if len(doctor['specialty']) > 8:
        # 將專長分組，降低閾值以更好地處理長專長列表
        specialty_groups = [doctor['specialty'][i:i+8] for i in range(0, len(doctor['specialty']), 8)]
        for i, group in enumerate(specialty_groups):
            specialty_text = f"{doctor['name']} 醫師專長：{' '.join(group)}"
            chunks.append({
                'text': specialty_text,
                'chunk_type': 'specialty',
                'group_index': i,
                'doctor_data': doctor
            })
    
    # 3. 現任職位片段
    if doctor['title']:
        title_text = f"{doctor['name']} 醫師現任職位：{' '.join(doctor['title'])}"
        chunks.append({
            'text': title_text,
            'chunk_type': 'current_position',
            'doctor_data': doctor
        })
    
    # 4. 經歷片段（按時間分組）
    if doctor['experience']:
        # 檢查是否有詳細的時間軸資訊
        has_timeline = any(re.search(r'\d{4}', exp) for exp in doctor['experience'])
        
        if has_timeline and len(doctor['experience']) > 6:
            # 按時間分組經歷
            experience_groups = [doctor['experience'][i:i+4] for i in range(0, len(doctor['experience']), 4)]
            for i, group in enumerate(experience_groups):
                exp_text = f"{doctor['name']} 醫師經歷：{' '.join(group)}"
                chunks.append({
                    'text': exp_text,
                    'chunk_type': 'experience',
                    'group_index': i,
                    'doctor_data': doctor
                })
        else:
            # 簡單經歷，統一處理
            exp_text = f"{doctor['name']} 醫師經歷：{' '.join(doctor['experience'])}"
            chunks.append({
                'text': exp_text,
                'chunk_type': 'experience',
                'doctor_data': doctor
            })
    
    # 5. 學歷片段
    if doctor['education']:
        edu_text = f"{doctor['name']} 醫師學歷：{' '.join(doctor['education'])}"
        chunks.append({
            'text': edu_text,
            'chunk_type': 'education',
            'doctor_data': doctor
        })
    
    # 6. 認證片段（如果認證很多）
    if len(doctor['certifications']) > 8:
        # 將認證分組，降低閾值以更好地處理長認證列表
        cert_groups = [doctor['certifications'][i:i+8] for i in range(0, len(doctor['certifications']), 8)]
        for i, group in enumerate(cert_groups):
            cert_text = f"{doctor['name']} 醫師專科認證：{' '.join(group)}"
            chunks.append({
                'text': cert_text,
                'chunk_type': 'certification',
                'group_index': i,
                'doctor_data': doctor
            })
    elif doctor['certifications']:
        # 簡單認證，統一處理
        cert_text = f"{doctor['name']} 醫師專科認證：{' '.join(doctor['certifications'])}"
        chunks.append({
            'text': cert_text,
            'chunk_type': 'certification',
            'doctor_data': doctor
        })
    
    # 為每個片段添加索引
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = i
        chunk['total_chunks'] = len(chunks)
    
    return chunks

def build_doctor_vector_db(data_source: str = "doctors3.json", 
                          custom_source_name: str = None,
                          collection_name: str = None,
                          db_path: str = "chroma_db",
                          use_smart_chunking: bool = False):
    """
    建立醫師向量資料庫
    
    Args:
        data_source: 資料來源檔案路徑
        custom_source_name: 自定義來源名稱
        collection_name: 自定義集合名稱
        db_path: 資料庫路徑
        use_smart_chunking: 是否使用智能切割
    """
    
    # 1. 載入醫師資料
    logger.info(f"載入醫師資料從: {data_source}")
    try:
        with open(data_source, 'r', encoding='utf-8') as f:
            doctors = json.load(f)
        logger.info(f"成功載入 {len(doctors)} 位醫師資料")
    except FileNotFoundError:
        logger.error(f"找不到 {data_source} 檔案")
        return
    except json.JSONDecodeError:
        logger.error("JSON 檔案格式錯誤")
        return
    
    # 2. 初始化處理器
    source_name = custom_source_name if custom_source_name else data_source
    processor = DoctorDataProcessor(data_source=source_name)
    
    # 3. 前處理醫師資料
    logger.info("開始前處理醫師資料...")
    processed_doctors = []
    skipped_count = 0
    
    for idx, doctor in enumerate(doctors):
        try:
            processed_doctor = processor.extract_key_info(doctor, idx)
            if processed_doctor:  # 只有當處理成功時才加入
                processed_doctors.append(processed_doctor)
                logger.info(f"已處理醫師 {idx+1}/{len(doctors)}: {processed_doctor['name']}")
            else:
                skipped_count += 1
                logger.warning(f"跳過醫師 {idx+1}/{len(doctors)}: {doctor.get('name', 'Unknown')} - 資料驗證失敗")
        except Exception as e:
            skipped_count += 1
            logger.error(f"處理醫師資料時發生錯誤: {doctor.get('name', 'Unknown')}, 錯誤: {e}")
            continue
    
    logger.info(f"處理完成: 成功 {len(processed_doctors)} 位醫師，跳過 {skipped_count} 位醫師")
    
    # 4. 初始化向量模型
    logger.info("初始化向量模型...")
    try:
        model = SentenceTransformer('BAAI/bge-m3')
        logger.info("向量模型載入成功")
    except Exception as e:
        logger.error(f"向量模型載入失敗: {e}")
        return
    
    # 5. 初始化 ChromaDB
    logger.info("初始化 ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # 根據是否使用智能切割設定不同的集合名稱
        if not collection_name:
            chunking_suffix = "smart" if use_smart_chunking else "basic"
            collection_name = f"doctors_{chunking_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.create_collection(collection_name)
        logger.info(f"ChromaDB 初始化成功，collection: {collection_name}")
    except Exception as e:
        logger.error(f"ChromaDB 初始化失敗: {e}")
        return
    
    # 6. 建立向量並存入資料庫
    if use_smart_chunking:
        logger.info("使用智能切割策略建立向量...")
        build_smart_chunked_vectors(processed_doctors, collection, model, processor)
    else:
        logger.info("使用基本策略建立向量...")
        build_basic_vectors(processed_doctors, collection, model, processor)
    
    # 7. 儲存處理後的資料
    try:
        chunking_suffix = "smart" if use_smart_chunking else "basic"
        output_filename = f'processed_doctors_{chunking_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_doctors, f, ensure_ascii=False, indent=2)
        logger.info(f"已儲存處理後的醫師資料到 {output_filename}")
    except Exception as e:
        logger.error(f"儲存處理後資料時發生錯誤: {e}")
    
    # 8. 建立統計報告
    logger.info("產生資料統計報告...")
    stats = generate_stats(processed_doctors, source_name, collection_name)
    
    # 9. 測試搜尋
    logger.info("進行搜尋測試...")
    test_search(collection, model, processor)
    
    logger.info("✅ 醫師向量庫建立完成！")

def test_doctor_processing():
    """測試醫師資料處理效果"""
    logger.info("開始測試醫師資料處理...")
    
    # 載入測試資料
    try:
        with open("doctorsv3.json", 'r', encoding='utf-8') as f:
            doctors = json.load(f)
    except FileNotFoundError:
        logger.error("找不到 doctorsv3.json 檔案")
        return
    
    processor = DoctorDataProcessor("doctorsv3.json")
    
    # 測試前5位醫師的處理
    for i, doctor in enumerate(doctors[:5]):
        logger.info(f"\n=== 測試醫師 {i+1}: {doctor.get('name', 'Unknown')} ===")
        
        # 測試資料驗證
        is_valid = processor.validate_doctor_data(doctor)
        logger.info(f"資料驗證: {'通過' if is_valid else '失敗'}")
        
        if is_valid:
            # 測試關鍵資訊提取
            processed = processor.extract_key_info(doctor, i)
            
            logger.info(f"處理後專長數量: {len(processed.get('specialty', []))}")
            logger.info(f"處理後職稱數量: {len(processed.get('title', []))}")
            logger.info(f"主要醫院: {processed.get('main_hospitals', [])}")
            logger.info(f"經驗年份: {processed.get('experience_years', {})}")
            
            # 測試搜尋文字生成
            search_text = processor.create_search_text(processed)
            logger.info(f"搜尋文字長度: {len(search_text)} 字符")
            
            # 測試智能切割
            chunks = chunk_doctor_data_smart(processed, processor)
            logger.info(f"智能切割片段數: {len(chunks)}")
            
            for j, chunk in enumerate(chunks[:3]):  # 只顯示前3個片段
                logger.info(f"  片段 {j+1} ({chunk['chunk_type']}): {chunk['text'][:100]}...")
    
    logger.info("\n=== 測試完成 ===")

def build_smart_chunked_vectors(processed_doctors: List[Dict], collection, model, processor):
    """使用智能切割策略建立向量"""
    total_chunks = 0
    
    for idx, doctor in enumerate(processed_doctors):
        try:
            # 使用智能切割
            chunks = chunk_doctor_data_smart(doctor, processor)
            
            logger.info(f"醫師 {doctor['name']} 切割為 {len(chunks)} 個片段")
            
            for chunk_info in chunks:
                try:
                    # 生成向量
                    embedding = model.encode(chunk_info['text'])
                    
                    # 準備元數據
                    metadata = {
                        'name': doctor['name'],
                        'department': doctor['department'],
                        'doctor_index': idx,
                        'chunk_id': chunk_info['chunk_id'],
                        'total_chunks': chunk_info['total_chunks'],
                        'chunk_type': chunk_info['chunk_type'],
                        'data_source': doctor['data_source'],
                        'import_date': doctor['import_date'],
                        'record_id': f"{doctor['record_id']}_chunk_{chunk_info['chunk_id']}"
                    }
                    
                    # 如果有分組資訊，添加到元數據
                    if 'group_index' in chunk_info:
                        metadata['group_index'] = chunk_info['group_index']
                    
                    # 存入 ChromaDB
                    collection.add(
                        documents=[chunk_info['text']],
                        embeddings=[embedding.tolist()],
                        metadatas=[metadata],
                        ids=[f"doctor_{idx}_chunk_{chunk_info['chunk_id']}"]
                    )
                    
                    total_chunks += 1
                    
                except Exception as e:
                    logger.error(f"處理片段時發生錯誤: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"切割醫師 {doctor['name']} 時發生錯誤: {e}")
            continue
    
    logger.info(f"共建立 {total_chunks} 個向量片段")

def build_basic_vectors(processed_doctors: List[Dict], collection, model, processor):
    """使用基本策略建立向量"""
    for idx, doctor in enumerate(processed_doctors):
        try:
            # 建立搜尋用文字和詳細描述
            search_text = processor.create_search_text(doctor)
            detailed_text = processor.create_detailed_text(doctor)
            
            # 生成向量
            embedding = model.encode(search_text)
            
            # 準備元數據
            metadata = {
                'name': doctor['name'],
                'department': doctor['department'],
                'specialty_count': len(doctor['specialty']),
                'title_count': len(doctor['title']),
                'main_hospitals': ','.join(doctor['main_hospitals']) if doctor['main_hospitals'] else '',
                'data_source': doctor['data_source'],
                'import_date': doctor['import_date'],
                'import_timestamp': doctor['import_timestamp'],
                'record_id': doctor['record_id'],
                'has_experience': 'yes' if doctor['experience'] else 'no',
                'has_education': 'yes' if doctor['education'] else 'no',
                'has_certifications': 'yes' if doctor['certifications'] else 'no',
                'experience_span': str(doctor.get('experience_years', {}).get('span', 0))
            }
            
            # 存入 ChromaDB
            collection.add(
                documents=[detailed_text],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[f"doctor_{idx}"]
            )
            
        except Exception as e:
            logger.error(f"處理醫師 {doctor['name']} 時發生錯誤: {e}")
            continue

def generate_stats(processed_doctors: List[Dict], source_name: str, collection_name: str) -> Dict:
    """生成統計報告"""
    stats = {
        'total_doctors': len(processed_doctors),
        'data_source': source_name,
        'collection_name': collection_name,
        'departments': Counter(d['department'] for d in processed_doctors),
        'hospitals': Counter(h for d in processed_doctors for h in d['main_hospitals']),
        'specialties': Counter(s for d in processed_doctors for s in d['specialty']),
        'avg_specialties_per_doctor': sum(len(d['specialty']) for d in processed_doctors) / len(processed_doctors),
        'avg_titles_per_doctor': sum(len(d['title']) for d in processed_doctors) / len(processed_doctors)
    }
    
    # 儲存統計報告
    try:
        stats_filename = f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        # 將 Counter 對象轉換為 dict
        stats_for_json = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
        
        with open(stats_filename, 'w', encoding='utf-8') as f:
            json.dump(stats_for_json, f, ensure_ascii=False, indent=2)
        logger.info(f"已儲存統計報告到 {stats_filename}")
        
        # 顯示統計資訊
        logger.info("=== 匯入統計 ===")
        logger.info(f"總醫師數: {stats['total_doctors']}")
        logger.info(f"科別分布: {dict(stats['departments'])}")
        logger.info(f"醫院分布: {dict(stats['hospitals'])}")
        logger.info(f"平均專長數: {stats['avg_specialties_per_doctor']:.2f}")

        
    except Exception as e:
        logger.error(f"儲存統計報告時發生錯誤: {e}")
    
    return stats

def test_search(collection, model, processor):
    """測試搜尋功能"""
    test_queries = [
        "心臟內科高血壓專家",
        "林宗憲醫師",
        "高醫附院心臟科主任",
        "心導管介入治療",
        "心臟超音波檢查",
        "肺動脈高壓治療",
        "心肌病變專家",
        "瓣膜性心臟病",
        "重症醫學專科醫師",
        "介入性心臟血管治療"
    ]
    
    for query in test_queries:
        try:
            test_embedding = model.encode(query)
            results = collection.query(
                query_embeddings=[test_embedding.tolist()],
                n_results=2,
                include=['metadatas', 'documents', 'distances']
            )
            
            logger.info(f"測試查詢: {query}")
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                logger.info(f"  結果 {i+1} (相似度: {1-distance:.3f}): {metadata['name']} - {doc[:80]}...")
            logger.info("---")
            
        except Exception as e:
            logger.error(f"測試查詢 '{query}' 時發生錯誤: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 執行測試模式
        test_doctor_processing()
    else:
        # 執行向量庫建立
        build_doctor_vector_db(
            data_source="doctorsv3.json",
            custom_source_name="高醫心臟科醫師名單v3",
            use_smart_chunking=True  # 啟用智能切割
        )
