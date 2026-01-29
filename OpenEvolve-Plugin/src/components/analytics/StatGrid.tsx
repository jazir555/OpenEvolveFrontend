import { MetricCard } from './MetricCard';
import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface Stat {
  title: string;
  value: number | string;
  change?: number;
  unit?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

interface StatGridProps {
  stats: Stat[];
  columns?: 2 | 3 | 4;
  className?: string;
}

function StatGridBase({ stats, columns = 4, className }: StatGridProps) {
  const gridCols = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={cn('stat-grid grid gap-4', gridCols[columns], className)}>
      {stats.map((stat, index) => (
        <MetricCard key={index} {...stat} />
      ))}
    </div>
  );
}

export const StatGrid = withComponentBoundary(StatGridBase, 'StatGrid');
