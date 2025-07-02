import React, { useState } from 'react';
import { Search, Send, Loader2 } from 'lucide-react';

interface QueryFormProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

const QueryForm: React.FC<QueryFormProps> = ({ onSubmit, isLoading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const exampleQueries = [
    '今天高醫心臟內科看到幾號？',
    '高醫急診室現在人多嗎？',
    '高醫門診時間表',
    '高醫掛號方式'
  ];

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* 查詢表單 */}
      <form onSubmit={handleSubmit} className="mb-8">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="輸入您的醫療資訊查詢，例如：今天高醫心臟內科看到幾號？"
            className="input-field pl-12 pr-16 text-lg"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="absolute inset-y-0 right-0 px-4 flex items-center bg-medical-600 hover:bg-medical-700 disabled:bg-gray-300 text-white rounded-r-lg transition-colors duration-200"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </div>
      </form>

      {/* 範例查詢 */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">💡 查詢範例：</h3>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuery(example)}
              disabled={isLoading}
              className="px-3 py-1.5 text-sm bg-white hover:bg-gray-50 text-medical-600 border border-medical-200 rounded-full transition-colors duration-200 disabled:opacity-50"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* 功能說明 */}
      <div className="bg-medical-50 border border-medical-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-medical-800 mb-2">🔍 系統功能：</h3>
        <ul className="text-sm text-medical-700 space-y-1">
          <li>• 即時搜尋 Google 獲取最新醫療資訊</li>
          <li>• 使用 GPT-4o 解析並提供自然語言回應</li>
          <li>• 支援醫院叫號、門診時間、急診狀況等查詢</li>
          <li>• 完全依賴網路即時資訊，無需資料庫</li>
        </ul>
      </div>
    </div>
  );
};

export default QueryForm; 
 