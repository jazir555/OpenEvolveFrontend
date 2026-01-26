/**
 * Benchmarks Route
 * Benchmark management and execution
 */

import { useState } from 'react';
import { BenchmarkRunner } from '../components/benchmark/BenchmarkRunner';
import { ResultsComparison } from '../components/benchmark/ResultsComparison';
import { DatasetUploader } from '../components/benchmark/DatasetUploader';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { QuickStats } from '../components/dashboard/QuickStats';

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

export default function BenchmarksPage() {
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [showUploader, setShowUploader] = useState(false);

  const handleDataParsed = (dataset: { records: unknown[] }) => {
    console.log('Dataset loaded:', dataset);
    setShowUploader(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Benchmarks
          </h1>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Compare workflow performance across multiple runs
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="secondary" onClick={() => setShowUploader(!showUploader)}>
            {showUploader ? 'Hide' : 'Upload Dataset'}
          </Button>
          <BenchmarkRunner />
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStats />

      {/* Dataset Uploader */}
      {showUploader && (
        <Card>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Upload Benchmark Dataset
          </h2>
          <DatasetUploader
            onDataParsed={handleDataParsed}
            onError={(error) => console.error('Dataset error:', error)}
          />
        </Card>
      )}

      {/* Results Comparison */}
      {results.length > 0 ? (
        <ResultsComparison results={results} />
      ) : (
        <Card>
          <div className="text-center py-12">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
              No benchmark results
            </h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Run a benchmark to see performance comparisons
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
