/**
 * Results View Component
 * Displays workflow execution results
 */

import { useWorkflowResults } from '../../hooks/use-workflows-api';
import { ExecutionResult } from '../../types/api';

interface ResultsViewProps {
  workflowId: string;
}

export function ResultsView({ workflowId }: ResultsViewProps) {
  const { data: results, isLoading, error } = useWorkflowResults(workflowId);

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-center">
          <svg className="mx-auto h-12 w-12 animate-spin text-blue-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Loading results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 dark:border-red-900 dark:bg-red-900/20">
        <p className="text-red-800 dark:text-red-400">
          Error loading results: {error.message}
        </p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="rounded-lg border border-dashed border-gray-300 p-12 text-center dark:border-gray-600">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          No results available yet
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Final Solution */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Final Solution
        </h3>
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <pre className="whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300">
            {results.final_solution}
          </pre>
        </div>
      </div>

      {/* Execution Statistics */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Execution Statistics
        </h3>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard
            label="Duration"
            value={`${Math.round(results.duration_seconds)}s`}
          />
          <StatCard
            label="Tokens Used"
            value={results.statistics.total_tokens_used.toLocaleString()}
          />
          <StatCard
            label="API Calls"
            value={results.statistics.total_api_calls.toLocaleString()}
          />
          <StatCard
            label="Success Rate"
            value={`${Math.round(results.statistics.success_rate * 100)}%`}
          />
        </div>
      </div>

      {/* Sub-problems */}
      {results.sub_problems && results.sub_problems.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Sub-problems ({results.sub_problems.length})
          </h3>
          <div className="space-y-4">
            {results.sub_problems.map((subProblem: any, index: number) => (
              <SubProblemCard key={subProblem.subproblem_id} subProblem={subProblem} index={index} />
            ))}
          </div>
        </div>
      )}

      {/* Export Buttons */}
      <div className="flex gap-2">
        <button className="inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">
          <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
          Export JSON
        </button>
        <button className="inline-flex items-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700">
          Export PDF
        </button>
        <button className="inline-flex items-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700">
          Copy to Clipboard
        </button>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="text-center">
      <p className="text-sm text-gray-600 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-gray-900 dark:text-white">{value}</p>
    </div>
  );
}

function SubProblemCard({ subProblem, index }: { subProblem: any; index: number }) {
  return (
    <details className="rounded-lg border border-gray-200 dark:border-gray-700">
      <summary className="cursor-pointer px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700">
        <div className="flex items-center justify-between">
          <span className="font-medium text-gray-900 dark:text-white">
            Sub-problem {index + 1}
          </span>
          <span className={`text-xs px-2 py-1 rounded-full ${
            subProblem.status === 'completed'
              ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
          }`}>
            {subProblem.status}
          </span>
        </div>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          {subProblem.problem}
        </p>
      </summary>
      <div className="border-t border-gray-200 px-4 py-3 dark:border-gray-700">
        <p className="text-sm text-gray-700 dark:text-gray-300">
          <strong className="text-gray-900 dark:text-white">Solution:</strong>
        </p>
        <pre className="mt-2 whitespace-pre-wrap text-xs text-gray-600 dark:text-gray-400">
          {subProblem.solution || 'No solution yet'}
        </pre>
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          Duration: {Math.round(subProblem.duration_seconds)}s
        </div>
      </div>
    </details>
  );
}
