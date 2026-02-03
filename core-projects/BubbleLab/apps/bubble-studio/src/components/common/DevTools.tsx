/**
 * DevTools Component
 * Development utilities for debugging
 */

import { useState } from 'react';

interface DevToolsProps {
  stores?: {
    name: string;
    state: unknown;
  }[];
  className?: string;
}

export function DevTools({ stores = [], className = '' }: DevToolsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<string>('state');

  if (process.env.NODE_ENV === 'production') {
    return null;
  }

  return (
    <div className={`fixed bottom-4 right-4 z-50 ${className}`}>
      {isOpen ? (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-gray-300 dark:border-gray-700 w-96 max-h-[600px] overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between p-3 border-b border-gray-300 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
              DevTools
            </h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              ✕
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-gray-300 dark:border-gray-700">
            <button
              onClick={() => setActiveTab('state')}
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === 'state'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              State
            </button>
            <button
              onClick={() => setActiveTab('storage')}
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === 'storage'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Storage
            </button>
            <button
              onClick={() => setActiveTab('actions')}
              className={`px-4 py-2 text-sm font-medium ${
                activeTab === 'actions'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Actions
            </button>
          </div>

          {/* Content */}
          <div className="p-4 overflow-auto max-h-[400px]">
            {activeTab === 'state' && (
              <div className="space-y-4">
                {stores.length === 0 ? (
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    No stores provided
                  </p>
                ) : (
                  stores.map((store) => (
                    <div key={store.name}>
                      <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
                        {store.name}
                      </h4>
                      <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-auto max-h-40">
                        {JSON.stringify(store.state, null, 2)}
                      </pre>
                    </div>
                  ))
                )}
              </div>
            )}

            {activeTab === 'storage' && (
              <div className="space-y-2">
                <button
                  onClick={() => localStorage.clear()}
                  className="w-full px-3 py-2 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400"
                >
                  Clear localStorage
                </button>
                <button
                  onClick={() => window.location.reload()}
                  className="w-full px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300"
                >
                  Reload Page
                </button>
              </div>
            )}

            {activeTab === 'actions' && (
              <div className="space-y-2">
                <button
                  onClick={() => {
                    const data = stores.map(s => ({ name: s.name, state: s.state }));
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'state-export.json';
                    a.click();
                  }}
                  className="w-full px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-400"
                >
                  Export State
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(stores.map(s => ({ name: s.name, state: s.state })), null, 2));
                    alert('State copied to clipboard');
                  }}
                  className="w-full px-3 py-2 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200 dark:bg-green-900/30 dark:text-green-400"
                >
                  Copy State to Clipboard
                </button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <button
          onClick={() => setIsOpen(true)}
          className="bg-gray-900 dark:bg-gray-700 text-white px-3 py-2 rounded-lg shadow-lg text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-600"
        >
          DevTools
        </button>
      )}
    </div>
  );
}
