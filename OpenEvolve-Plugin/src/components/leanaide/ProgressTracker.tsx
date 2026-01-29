import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface ProgressStep {
  id: string;
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  timestamp?: string;
  duration?: number;
}

interface ProgressTrackerProps {
  steps: ProgressStep[];
  className?: string;
}

function ProgressTrackerBase({ steps, className }: ProgressTrackerProps) {
  const getStepIcon = (status: ProgressStep['status']) => {
    switch (status) {
      case 'completed':
        return '✓';
      case 'failed':
        return '✕';
      case 'in_progress':
        return '⟳';
      case 'pending':
        return '○';
    }
  };

  const getStepColor = (status: ProgressStep['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 border-green-500 text-green-700';
      case 'failed':
        return 'bg-red-100 border-red-500 text-red-700';
      case 'in_progress':
        return 'bg-blue-100 border-blue-500 text-blue-700 animate-pulse';
      case 'pending':
        return 'bg-gray-100 border-gray-300 text-gray-500';
    }
  };

  const getConnectorColor = (index: number) => {
    if (index === steps.length - 1) return 'border-gray-300';
    const nextStep = steps[index + 1];
    if (nextStep.status === 'completed') return 'border-green-500';
    if (nextStep.status === 'failed') return 'border-red-500';
    return 'border-gray-300';
  };

  return (
    <div className={cn('progress-tracker', className)}>
      <div className="relative">
        {steps.map((step, index) => (
          <div key={step.id} className="relative flex items-start gap-4 pb-8 last:pb-0">
            {/* Vertical Line */}
            {index < steps.length - 1 && (
              <div
                className={cn(
                  'absolute left-[15px] top-8 w-0.5 h-full',
                  getConnectorColor(index)
                )}
              />
            )}

            {/* Icon */}
            <div className={cn(
              'relative z-10 w-8 h-8 rounded-full border-2 flex items-center justify-center text-sm font-medium flex-shrink-0',
              getStepColor(step.status)
            )}>
              {getStepIcon(step.status)}
            </div>

            {/* Content */}
            <div className="flex-1 pt-1">
              <div className="flex items-center justify-between mb-1">
                <h4 className={cn(
                  'text-sm font-medium',
                  step.status === 'completed' && 'text-green-900',
                  step.status === 'failed' && 'text-red-900',
                  step.status === 'in_progress' && 'text-blue-900',
                  step.status === 'pending' && 'text-gray-600'
                )}>
                  {step.label}
                </h4>
                {step.duration && (
                  <span className="text-xs text-gray-500">
                    {step.duration}ms
                  </span>
                )}
              </div>

              {step.timestamp && (
                <div className="text-xs text-gray-500 mb-1">
                  {(() => {
                    const date = new Date(step.timestamp);
                    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
                  })()}
                </div>
              )}

              {step.status === 'in_progress' && (
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div className="bg-blue-600 h-1.5 rounded-full animate-pulse w-1/2" />
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export const ProgressTracker = withComponentBoundary(ProgressTrackerBase, 'ProgressTracker');
