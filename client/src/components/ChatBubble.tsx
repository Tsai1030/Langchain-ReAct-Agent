import React from 'react';

export default function ChatBubble({ role, content, timestamp }: { role: 'user' | 'assistant', content: string, timestamp?: string }) {
  const isUser = role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-2`}>
      <div className={`max-w-[70%] rounded-xl px-4 py-2 shadow
        ${isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-white text-gray-900 rounded-bl-none border border-gray-200'}`}>
        <div className="whitespace-pre-wrap">{content}</div>
        {timestamp && <div className="text-xs text-gray-400 mt-1 text-right">{timestamp}</div>}
      </div>
    </div>
  );
} 