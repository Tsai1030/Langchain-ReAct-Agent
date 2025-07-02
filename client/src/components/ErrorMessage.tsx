import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorMessageProps {
  error: string;
  onRetry?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ error, onRetry }) => {
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="card border-red-200 bg-red-50">
        <div className="flex items-start space-x-3">
          <AlertCircle className="h-6 w-6 text-red-500 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="text-lg font-medium text-red-800 mb-2">
              查詢失敗
            </h3>
            <p className="text-red-700 mb-4">
              {error}
            </p>
            
            {/* 常見問題解決方案 */}
            <div className="bg-white rounded-lg p-3 mb-4">
              <h4 className="text-sm font-medium text-gray-900 mb-2">
                💡 可能的解決方案：
              </h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• 檢查網路連線是否正常</li>
                <li>• 確認查詢內容是否清楚明確</li>
                <li>• 稍後再試，可能是暫時性服務問題</li>
                <li>• 嘗試使用不同的查詢方式</li>
              </ul>
            </div>

            {/* 重試按鈕 */}
            {onRetry && (
              <button
                onClick={onRetry}
                className="btn-primary inline-flex items-center space-x-2"
              >
                <RefreshCw className="h-4 w-4" />
                <span>重新查詢</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage; 