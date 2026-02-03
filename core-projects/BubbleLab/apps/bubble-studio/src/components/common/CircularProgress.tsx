/**
 * CircularProgress Component
 * Circular progress indicator
 */

interface CircularProgressProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  showLabel?: boolean;
  className?: string;
}

export function CircularProgress({
  value,
  max = 100,
  size = 40,
  strokeWidth = 4,
  variant = 'default',
  showLabel = false,
  className = '',
}: CircularProgressProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  const variantStyles = {
    default: 'stroke-blue-600',
    success: 'stroke-green-600',
    warning: 'stroke-yellow-600',
    danger: 'stroke-red-600',
  };

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-gray-200 dark:text-gray-700"
        />

        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={variantStyles[variant]}
          style={{
            transition: 'stroke-dashoffset 0.3s ease-out',
          }}
        />
      </svg>

      {showLabel && (
        <span className="absolute text-xs font-medium text-gray-900 dark:text-white">
          {Math.round(percentage)}%
        </span>
      )}
    </div>
  );
}
