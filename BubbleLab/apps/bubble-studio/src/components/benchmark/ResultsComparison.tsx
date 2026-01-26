/**
 * ResultsComparison Component
 * Compare benchmark results across workflows
 */

import { Card } from '../common/Card';

interface BenchmarkResult {
  workflowId: string;
  workflowName: string;
  iteration: number;
  duration: number;
  success: boolean;
  error?: string;
  metrics?: {
    tokensUsed: number;
    cost: number;
    quality?: number;
  };
}

interface ComparisonData {
  results: BenchmarkResult[];
  averageDurations: Record<string, number>;
  successRates: Record<string, number>;
  totalCosts: Record<string, number>;
}

export function ResultsComparison({ results }: { results: BenchmarkResult[] }) {
  const comparisonData = analyzeResults(results);

  return (
    <div className="space-y-6">
      {/* Duration Comparison */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Average Duration (seconds)
        </h3>
        <div className="space-y-3">
          {Object.entries(comparisonData.averageDurations).map(
            ([workflowName, duration]) => (
              <div key={workflowName}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300">
                    {workflowName}
                  </span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {duration.toFixed(2)}s
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{
                      width: `${(duration / Math.max(...Object.values(comparisonData.averageDurations))) * 100}%`,
                    }}
                  />
                </div>
              </div>
            )
          )}
        </div>
      </Card>

      {/* Success Rate Comparison */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Success Rate
        </h3>
        <div className="space-y-3">
          {Object.entries(comparisonData.successRates).map(
            ([workflowName, rate]) => (
              <div key={workflowName}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300">
                    {workflowName}
                  </span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {(rate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      rate > 0.9
                        ? 'bg-green-600'
                        : rate > 0.7
                        ? 'bg-yellow-600'
                        : 'bg-red-600'
                    }`}
                    style={{ width: `${rate * 100}%` }}
                  />
                </div>
              </div>
            )
          )}
        </div>
      </Card>

      {/* Cost Comparison */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Total Cost (USD)
        </h3>
        <div className="space-y-3">
          {Object.entries(comparisonData.totalCosts).map(
            ([workflowName, cost]) => (
              <div key={workflowName} className="flex justify-between items-center">
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {workflowName}
                </span>
                <span className="font-medium text-gray-900 dark:text-white">
                  ${cost.toFixed(4)}
                </span>
              </div>
            )
          )}
        </div>
      </Card>

      {/* Detailed Results Table */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Detailed Results
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                  Workflow
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                  Iteration
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                  Duration
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                  Status
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                  Cost
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {results.map((result, index) => (
                <tr key={index}>
                  <td className="px-4 py-2 text-sm text-gray-900 dark:text-white">
                    {result.workflowName}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400">
                    {result.iteration}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400">
                    {result.duration.toFixed(2)}s
                  </td>
                  <td className="px-4 py-2">
                    {result.success ? (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                        Success
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400">
                        Failed
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400">
                    ${result.metrics?.cost.toFixed(4) || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function analyzeResults(results: BenchmarkResult[]): ComparisonData {
  const workflowResults: Record<string, BenchmarkResult[]> = {};

  // Group by workflow
  results.forEach((result) => {
    if (!workflowResults[result.workflowName]) {
      workflowResults[result.workflowName] = [];
    }
    workflowResults[result.workflowName].push(result);
  });

  // Calculate averages
  const averageDurations: Record<string, number> = {};
  const successRates: Record<string, number> = {};
  const totalCosts: Record<string, number> = {};

  Object.entries(workflowResults).forEach(([workflowName, workflowResults]) => {
    // Average duration
    const totalDuration = workflowResults.reduce(
      (sum, r) => sum + r.duration,
      0
    );
    averageDurations[workflowName] = totalDuration / workflowResults.length;

    // Success rate
    const successCount = workflowResults.filter((r) => r.success).length;
    successRates[workflowName] = successCount / workflowResults.length;

    // Total cost
    totalCosts[workflowName] = workflowResults.reduce(
      (sum, r) => sum + (r.metrics?.cost || 0),
      0
    );
  });

  return {
    results,
    averageDurations,
    successRates,
    totalCosts,
  };
}
