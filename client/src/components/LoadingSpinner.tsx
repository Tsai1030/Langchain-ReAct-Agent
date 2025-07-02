import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = '正在搜尋醫療資訊...' 
}) => {
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="card text-center">
        <div className="flex flex-col items-center space-y-4">
          {/* 載入動畫 */}
          <div className="relative">
            <div className="w-12 h-12 border-4 border-medical-200 border-t-medical-600 rounded-full animate-spin"></div>
            <div className="absolute inset-0 w-12 h-12 border-4 border-transparent border-t-medical-400 rounded-full animate-ping"></div>
          </div>
          
          {/* 載入訊息 */}
          <div className="space-y-2">
            <h3 className="text-lg font-medium text-gray-900">{message}</h3>
            <p className="text-sm text-gray-600">
              正在透過 Google 搜尋並使用 AI 分析結果...
            </p>
          </div>

          {/* 載入步驟指示器 */}
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-medical-500 rounded-full animate-pulse"></div>
              <span>搜尋中</span>
            </div>
            <div className="w-4 h-px bg-gray-300"></div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-medical-500 rounded-full animate-pulse" style={{ animationDelay: '0.5s' }}></div>
              <span>分析中</span>
            </div>
            <div className="w-4 h-px bg-gray-300"></div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-gray-300 rounded-full"></div>
              <span>生成回應</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner; 