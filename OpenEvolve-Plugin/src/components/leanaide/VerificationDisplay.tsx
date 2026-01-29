import { useState } from 'react';
import { cn } from '@/lib/utils';
import { BubbleBadge, BubbleButton, BubbleCard } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface VerificationError {
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export interface VerificationResult {
  status: 'pending' | 'running' | 'success' | 'failed';
  errors: VerificationError[];
  warnings: VerificationError[];
  output: string;
  duration?: number;
}

interface VerificationDisplayProps {
  result: VerificationResult;
  className?: string;
}

function VerificationDisplayBase({ result, className }: VerificationDisplayProps) {
  const [selectedTab, setSelectedTab] = useState<'errors' | 'output' | 'details'>('errors');

  const getStatusTone = (status: VerificationResult['status']) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'failed':
        return 'danger';
      case 'running':
        return 'info';
      case 'pending':
      default:
        return 'neutral';
    }
  };

  const errors = result.errors || [];
  const warnings = result.warnings || [];
  const allErrorsAndWarnings = [...errors, ...warnings];

  return (
    <BubbleCard className={cn('verification-display', className)} title="Verification Status">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <BubbleBadge tone={getStatusTone(result.status)}>
              {result.status.toUpperCase()}
            </BubbleBadge>
            {result.duration && (
              <span className="text-sm text-gray-600">
                Completed in {result.duration}ms
              </span>
            )}
          </div>

          {result.status === 'failed' && (
            <div className="text-sm text-gray-600">
              {errors.length} errors, {warnings.length} warnings
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {(['errors', 'output', 'details'] as const).map((tab) => {
            const count =
              tab === 'errors' ? allErrorsAndWarnings.length :
              tab === 'output' ? (result.output ? 1 : 0) :
              1;

            return (
              <BubbleButton
                key={tab}
                onClick={() => setSelectedTab(tab)}
                disabled={count === 0}
                variant={selectedTab === tab ? 'primary' : 'secondary'}
              >
                {tab} {count > 0 && `(${count})`}
              </BubbleButton>
            );
          })}
        </div>

        <div>
          {selectedTab === 'errors' && (
            <div className="space-y-2">
              {allErrorsAndWarnings.length === 0 ? (
                <div className="text-center py-6 text-gray-500">
                  <p className="text-emerald-600 font-medium">No errors or warnings</p>
                </div>
              ) : (
                allErrorsAndWarnings.map((error, index) => (
                  <div
                    key={index}
                    className={cn(
                      'flex items-start gap-3 rounded-lg border p-3',
                      error.severity === 'error' && 'border-red-200 bg-red-50',
                      error.severity === 'warning' && 'border-yellow-200 bg-yellow-50',
                      error.severity === 'info' && 'border-blue-200 bg-blue-50'
                    )}
                  >
                    <BubbleBadge
                      tone={error.severity === 'error' ? 'danger' : error.severity === 'warning' ? 'warning' : 'info'}
                    >
                      {error.severity.toUpperCase()}
                    </BubbleBadge>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900">
                        {error.message}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Line {error.line}, Column {error.column}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {selectedTab === 'output' && (
            <div>
              {result.output ? (
                <pre className="bg-gray-50 p-4 rounded-lg text-sm font-mono overflow-x-auto whitespace-pre-wrap">
                  {result.output}
                </pre>
              ) : (
                <div className="text-center py-6 text-gray-500">No output available</div>
              )}
            </div>
          )}

          {selectedTab === 'details' && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="font-medium text-gray-700">Status</div>
                  <div className="text-gray-900">{result.status}</div>
                </div>
                {result.duration && (
                  <div>
                    <div className="font-medium text-gray-700">Duration</div>
                    <div className="text-gray-900">{result.duration}ms</div>
                  </div>
                )}
                <div>
                  <div className="font-medium text-gray-700">Errors</div>
                  <div className="text-gray-900">{errors.length}</div>
                </div>
                <div>
                  <div className="font-medium text-gray-700">Warnings</div>
                  <div className="text-gray-900">{warnings.length}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </BubbleCard>
  );
}

export const VerificationDisplay = withComponentBoundary(
  VerificationDisplayBase,
  'VerificationDisplay'
);
