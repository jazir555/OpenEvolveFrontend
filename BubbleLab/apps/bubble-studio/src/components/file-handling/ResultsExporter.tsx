/**
 * Results Exporter Component
 * Export workflow results in various formats
 */

import { ExecutionResult } from '../../types/api';
import { useState } from 'react';

interface ResultsExporterProps {
  results: ExecutionResult;
}

export function ResultsExporter({ results }: ResultsExporterProps) {
  const [isExporting, setIsExporting] = useState(false);

  const exportAsJSON = () => {
    const data = JSON.stringify(results, null, 2);
    downloadFile(data, 'workflow-results.json', 'application/json');
  };

  const exportAsText = () => {
    let text = `Workflow Results\n${'='.repeat(50)}\n\n`;
    text += `Final Solution:\n${results.final_solution}\n\n`;
    text += `Duration: ${results.duration_seconds}s\n`;
    text += `Tokens Used: ${results.statistics.total_tokens_used}\n`;
    text += `API Calls: ${results.statistics.total_api_calls}\n\n`;

    results.sub_problems.forEach((sub, i) => {
      text += `Sub-problem ${i + 1}:\n`;
      text += `Problem: ${sub.problem}\n`;
      text += `Solution: ${sub.solution}\n`;
      text += `Duration: ${sub.duration_seconds}s\n\n`;
    });

    downloadFile(text, 'workflow-results.txt', 'text/plain');
  };

  const exportAsPDF = async () => {
    // Placeholder for PDF export
    // Would need a library like jsPDF
    alert('PDF export not yet implemented');
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(results.final_solution);
      alert('Solution copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const downloadFile = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        Export Results
      </h3>

      <div className="grid gap-3 sm:grid-cols-2">
        <button
          onClick={exportAsJSON}
          disabled={isExporting}
          className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
          Export as JSON
        </button>

        <button
          onClick={exportAsText}
          disabled={isExporting}
          className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z"
              clipRule="evenodd"
            />
          </svg>
          Export as Text
        </button>

        <button
          onClick={exportAsPDF}
          disabled={isExporting}
          className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm5 6a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V8z"
              clipRule="evenodd"
            />
          </svg>
          Export as PDF
        </button>

        <button
          onClick={copyToClipboard}
          disabled={isExporting}
          className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
            <path
              d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3zM5 5a3 3 0 013-3h4a3 3 0 013 3v1H5V5z"
            />
          </svg>
          Copy to Clipboard
        </button>
      </div>
    </div>
  );
}
