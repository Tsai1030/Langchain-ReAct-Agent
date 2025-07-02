// API 回應型別
export interface QueryResponse {
  success: boolean;
  query: string;
  response: string;
  searchResults: {
    organic: SearchResult[];
    totalResults: number;
  };
  timestamp: string;
}

// 搜尋結果型別
export interface SearchResult {
  title: string;
  link: string;
  snippet: string;
  position: number;
}

// 查詢狀態型別
export type QueryStatus = 'idle' | 'loading' | 'success' | 'error';

// 查詢歷史記錄型別
export interface QueryHistory {
  id: string;
  query: string;
  response?: string;
  timestamp: string;
  status: QueryStatus;
  searchResults?: {
    organic: SearchResult[];
    totalResults: number;
  };
  // 多輪對話用
  role?: 'user' | 'assistant';
  content?: string;
}

// 錯誤回應型別
export interface ErrorResponse {
  error: string;
  message: string;
} 