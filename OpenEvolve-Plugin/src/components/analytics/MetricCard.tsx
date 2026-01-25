import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

interface MetricCardProps {
  title: string;
  value: number | string;
  change?: number;
  unit?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

function MetricCardBase({
  title,
  value,
  change,
  unit,
  icon,
  trend,
  className,
}: MetricCardProps) {
  const getTrendColor = () => {
    if (trend === 'up') return 'text-green-600';
    if (trend === 'down') return 'text-red-600';
    return 'text-gray-600';
  };

  const getTrendIcon = () => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  return (
    <div className={cn('metric-card bg-white border border-gray-200 rounded-lg p-6', className)}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-bold text-gray-900">
              {typeof value === 'number' ? value.toLocaleString() : value}
              {unit && <span className="text-lg font-normal text-gray-600 ml-1">{unit}</span>}
            </div>
          </div>
        </div>
        {icon && <div className="text-gray-400">{icon}</div>}
      </div>

      {change !== undefined && (
        <div className={cn('flex items-center gap-1 text-sm font-medium', getTrendColor())}>
          <span>{getTrendIcon()}</span>
          <span>{Math.abs(change)}%</span>
          <span className="text-gray-500 font-normal">from last period</span>
        </div>
      )}
    </div>
  );
}

export const MetricCard = withComponentBoundary(MetricCardBase, 'MetricCard');
