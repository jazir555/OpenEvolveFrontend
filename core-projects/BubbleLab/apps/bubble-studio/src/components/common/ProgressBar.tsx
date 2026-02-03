/**
 * ProgressBar Component
 * Linear progress bar
 */

interface ProgressBarProps {
  progress: number;
  color?: 'blue' | 'green' | 'yellow' | 'red';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export function ProgressBar({
  progress,
  color = 'blue',
  size = 'md',
  showLabel = false
}: ProgressBarProps) {
  const sizeStyles = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const colorStyles = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600',
  };

  const clampedProgress = Math.min(100, Math.max(0, progress));

  return (
    <div>
      {showLabel && (
        <div className="mb-1 flex items-center justify-between text-sm">
          <span className="text-gray-700 dark:text-gray-300">Progress</span>
          <span className="font-medium text-gray-900 dark:text-white">{Math.round(clampedProgress)}%</span>
        </div>
      )}
      <div className={`w-full rounded-full bg-gray-200 dark:bg-gray-700 ${sizeStyles[size]}`}>
        <div
          className={`${colorStyles[color]} rounded-full transition-all duration-300`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
    </div>
  );
}
