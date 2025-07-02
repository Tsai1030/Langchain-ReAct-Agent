import axios from 'axios';
import { QueryResponse, ErrorResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// 建立 axios 實例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 秒超時
  headers: {
    'Content-Type': 'application/json',
  },
});

// 請求攔截器
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🚀 API 請求: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API 請求錯誤:', error);
    return Promise.reject(error);
  }
);

// 回應攔截器
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API 回應: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API 回應錯誤:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// 醫療資訊查詢 API
export const queryMedicalInfo = async (query: string, history: any[]): Promise<QueryResponse> => {
  try {
    const response = await apiClient.post<QueryResponse>('/query', { query, history });
    return response.data;
  } catch (error: any) {
    if (error.response?.data) {
      throw new Error(error.response.data.message || error.response.data.error);
    }
    throw new Error(error.message || '網路連線錯誤');
  }
};

// 健康檢查 API
export const checkHealth = async (): Promise<{ status: string; timestamp: string }> => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error: any) {
    throw new Error('伺服器連線失敗');
  }
};

export async function askAgent(question: string): Promise<string> {
  const res = await fetch("http://localhost:8000/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await res.json();
  return data.result;
}

export default apiClient; 