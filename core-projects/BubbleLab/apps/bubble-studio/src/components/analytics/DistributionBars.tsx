/**
 * DistributionBars Component
 * Lightweight horizontal CSS bar chart for key/value distributions
 * (e.g. artifact_type_distribution, domain_distribution).
 */

import { Card } from '../common/Card';

interface DistributionBarsProps {
  data: Record<string, number>;
  title?: string;
  subtitle?: string;
  className?: string;
  max?: number;
  formatValue?: (value: number) => string;
  emptyMessage?: string;
}

const barColors = [
  'bg-blue-600',
  'bg-green-600',
  'bg-yellow-600',
  'bg-purple-600',
  'bg-red-600',
  'bg-indigo-600',
  'bg-pink-600',
  'bg-teal-600',
];

function formatNumber(value: number): string {
  return value.toLocaleString();
}

export function DistributionBars({
  data,
  title,
  subtitle,
  className = '',
  max,
  formatValue = formatNumber,
  emptyMessage = 'No data available',
}: DistributionBarsProps) {
  const entries = Object.entries(data).filter(
    ([, value]) => typeof value === 'number' && !Number.isNaN(value)
  );

  const safeMax = max ?? (entries.length > 0 ? Math.max(...entries.map(([, v]) => v)) : 1);
  const scaleMax = safeMax > 0 ? safeMax : 1;

  return (
    <Card title={title} subtitle={subtitle} className={className}>
      {entries.length === 0 ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">{emptyMessage}</p>
      ) : (
        <div className="space-y-3">
          {entries.map(([label, value], index) => {
            const percentage = (value / scaleMax) * 100;
            const color = barColors[index % barColors.length];
            return (
              <div key={label} className="flex items-center gap-3">
                <span
                  className="w-36 flex-shrink-0 truncate text-sm text-gray-600 dark:text-gray-400"
                  title={label}
                >
                  {label}
                </span>
                <div className="h-3 flex-1 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                  <div
                    className={`h-full rounded-full ${color} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span className="w-16 flex-shrink-0 text-right text-sm font-medium text-gray-900 dark:text-white">
                  {formatValue(value)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
