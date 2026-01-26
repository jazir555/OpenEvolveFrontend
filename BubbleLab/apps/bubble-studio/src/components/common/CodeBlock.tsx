/**
 * CodeBlock Component
 * Syntax highlighted code display
 */

import { useState } from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  onCopy?: () => void;
}

export function CodeBlock({ code, language = 'typescript', filename, onCopy }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    onCopy?.();
  };

  return (
    <div className="rounded-lg bg-gray-900 dark:bg-black">
      {filename && (
        <div className="flex items-center justify-between border-b border-gray-700 px-4 py-2">
          <span className="text-sm text-gray-400">{filename}</span>
          <span className="text-xs text-gray-500">{language}</span>
        </div>
      )}
      <pre className="overflow-x-auto p-4">
        <code className="text-sm text-gray-100">{code}</code>
      </pre>
      <div className="flex items-center justify-end border-t border-gray-700 px-4 py-2">
        <button
          onClick={handleCopy}
          className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
        >
          {copied ? (
            <>
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010-1.414l-8-8a1 1 0 00-1.414 0l-8 8a1 1 0 001.414 0l8-8z"
                  clipRule="evenodd"
                />
              </svg>
              Copied!
            </>
          ) : (
            <>
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8 3a1 1 0 011-1v6a1 1 0 01-1 1v2a1 1 0 001 1h6a1 1 0 001-1v-2a1 1 0 01-1-1V4a1 1 0 011-1z" />
                <path d="M6 8a1 1 0 011 1v6a1 1 0 001 1h2a1 1 0 001-1V8a1 1 0 00-1-1z" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
    </div>
  );
}
