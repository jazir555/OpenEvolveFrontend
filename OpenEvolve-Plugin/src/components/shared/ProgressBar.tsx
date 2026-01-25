import { cn } from '@/utils/helpers';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

interface ProgressBarProps {
  progress: number;
  label?: string;
  showPercentage?: boolean;
  className?: string;
  color?: 'blue' | 'green' | 'yellow' | 'red';
}

function ProgressBarBase({
  progress,
  label,
  showPercentage = true,
  className,
  color = 'blue',
}: ProgressBarProps) {
  const colorClasses = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600',
  };

  const clampedProgress = Math.min(100, Math.max(0, progress));

  return (
    <div className={cn('progress-bar w-full', className)}>
      {label && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">{label}</span>
          {showPercentage && (
            <span className="text-sm font-semibold text-gray-900">
              {Math.round(clampedProgress)}%
            </span>
          )}
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
        <div
          className={cn(
            'h-2.5 rounded-full transition-all duration-300 ease-out',
            colorClasses[color]
          )}
          style={{ width: `${clampedProgress}%` }}
          role="progressbar"
          aria-valuenow={clampedProgress}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}

export const ProgressBar = withComponentBoundary(ProgressBarBase, 'ProgressBar');
