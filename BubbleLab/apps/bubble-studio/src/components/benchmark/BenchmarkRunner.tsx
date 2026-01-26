/**
 * Benchmark Runner Component
 * Run multiple workflows and compare results
 */

import { useState } from 'react';
import { useWorkflows } from '../../hooks/use-workflows-api';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Select } from '../common/Select';

export function BenchmarkRunner() {
  const { data: workflows } = useWorkflows();
  const [selectedWorkflows, setSelectedWorkflows] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<any>(null);

  const workflowOptions = (workflows || []).map((w) => ({
    value: w.id,
    label: w.name,
  }));

  const handleRunBenchmark = async () => {
    if (selectedWorkflows.length === 0) return;

    setIsRunning(true);
    // Simulate benchmark execution
    await new Promise(resolve => setTimeout(resolve, 3000));

    setResults({
      completed: true,
      results: selectedWorkflows.map((workflowId) => {
        const workflow = workflows?.find((w) => w.id === workflowId);
        return {
          workflowId,
          workflowName: workflow?.name,
          duration: Math.random() * 100 + 20,
          tokens: Math.floor(Math.random() * 10000 + 1000),
          success: Math.random() > 0.2,
        };
      }),
    });

    setIsRunning(false);
  };

  return (
    <div className="space-y-6">
      <Card title="Benchmark Runner" subtitle="Compare workflow performance">
        <div className="space-y-4">
          {/* Workflow Selection */}
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
              Select Workflows
            </label>
            <div className="space-y-2">
              {workflowOptions.map((option) => (
                <label key={option.value} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selectedWorkflows.includes(option.value)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedWorkflows([...selectedWorkflows, option.value]);
                      } else {
                        setSelectedWorkflows(selectedWorkflows.filter((id) => id !== option.value));
                      }
                    }}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-900 dark:text-white">{option.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <Button
              onClick={handleRunBenchmark}
              disabled={selectedWorkflows.length === 0 || isRunning}
              isLoading={isRunning}
            >
              Run Benchmark
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setSelectedWorkflows([]);
                setResults(null);
              }}
            >
              Clear
            </Button>
          </div>
        </div>
      </Card>

      {/* Results */}
      {results && (
        <Card title="Results">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase dark:text-gray-400">
                    Workflow
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase dark:text-gray-400">
                    Duration (s)
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase dark:text-gray-400">
                    Tokens
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase dark:text-gray-400">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {results.results.map((result: any, index: number) => (
                  <tr key={index}>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-white">
                      {result.workflowName}
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400">
                      {result.duration.toFixed(2)}
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400">
                      {result.tokens.toLocaleString()}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className={`inline-flex rounded-full px-2 text-xs font-medium ${
                          result.success
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                        }`}
                      >
                        {result.success ? 'Success' : 'Failed'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
