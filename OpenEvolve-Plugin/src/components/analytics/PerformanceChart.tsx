import React from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface PerformanceDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

interface PerformanceChartProps {
  data: PerformanceDataPoint[];
  type?: 'line' | 'bar';
  title?: string;
  color?: string;
  className?: string;
  height?: number;
}

function PerformanceChartBase({
  data,
  type = 'line',
  title,
  color = '#3b82f6',
  className,
  height = 300,
}: PerformanceChartProps) {
  const ChartComponent = type === 'line' ? LineChart : BarChart;

  return (
    <div className={cn('performance-chart bg-white border border-gray-200 rounded-lg p-6', className)}>
      {title && <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="timestamp"
            tick={{ fill: '#6b7280', fontSize: 12 }}
            stroke="#9ca3af"
          />
          <YAxis
            tick={{ fill: '#6b7280', fontSize: 12 }}
            stroke="#9ca3af"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '0.5rem',
            }}
          />
          <Legend />
          {type === 'line' ? (
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              name={title || 'Value'}
            />
          ) : (
            <Bar
              type="monotone"
              dataKey="value"
              fill={color}
              name={title || 'Value'}
            />
          )}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
}

export const PerformanceChart = withComponentBoundary(PerformanceChartBase, 'PerformanceChart');
