/**
 * Charts Component
 * Data visualization components using CSS/SVG
 */

import { Card } from '../common/Card';

interface ChartData {
  label: string;
  value: number;
  color?: string;
}

// Simple bar chart using CSS
interface BarChartProps {
  data: ChartData[];
  title?: string;
  height?: number;
  horizontal?: boolean;
  className?: string;
}

export function BarChart({
  data,
  title,
  height = 200,
  horizontal = false,
  className = '',
}: BarChartProps) {
  const maxValue = Math.max(1, ...data.map((d) => d.value));

  return (
    <Card className={className}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          {title}
        </h3>
      )}

      <div
        className={horizontal ? 'space-y-2' : 'flex items-end gap-2'}
        style={{ height: horizontal ? 'auto' : `${height}px` }}
      >
        {data.map((item, index) => {
          const percentage = (item.value / maxValue) * 100;
          const color = item.color || 'bg-blue-600';

          if (horizontal) {
            return (
              <div key={index} className="flex items-center gap-2">
                <span className="text-sm text-gray-600 dark:text-gray-400 w-24 text-right">
                  {item.label}
                </span>
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full ${color} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-white w-12">
                  {item.value}
                </span>
              </div>
            );
          }

          return (
            <div key={index} className="flex-1 flex flex-col items-center gap-2">
              <div
                className={`w-full ${color} rounded-t transition-all duration-500`}
                style={{ height: `${percentage}%`, minHeight: '4px' }}
              />
              <span className="text-xs text-gray-600 dark:text-gray-400 text-center w-full truncate">
                {item.label}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// Simple pie chart using SVG
interface PieChartProps {
  data: ChartData[];
  title?: string;
  size?: number;
  className?: string;
}

export function PieChart({
  data,
  title,
  size = 200,
  className = '',
}: PieChartProps) {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const safeTotal = total === 0 ? 1 : total;
  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

  let currentAngle = 0;
  const slices = data.map((item, index) => {
    const percentage = (item.value / safeTotal) * 100;
    const angle = (item.value / safeTotal) * 360;

    const slice = {
      ...item,
      percentage,
      angle,
      startAngle: currentAngle,
      endAngle: currentAngle + angle,
      color: item.color || colors[index % colors.length],
    };

    currentAngle += angle;
    return slice;
  });

  // Calculate SVG path for pie slice
  const getSlicePath = (startAngle: number, endAngle: number, radius: number) => {
    const start = polarToCartesian(radius, radius, radius, endAngle);
    const end = polarToCartesian(radius, radius, radius, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1';

    return [
      'M',
      radius,
      radius,
      'L',
      start.x,
      start.y,
      'A',
      radius,
      radius,
      0,
      largeArcFlag,
      0,
      end.x,
      end.y,
      'Z',
    ].join(' ');
  };

  const polarToCartesian = (
    centerX: number,
    centerY: number,
    radius: number,
    angleInDegrees: number
  ) => {
    const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians),
    };
  };

  return (
    <Card className={className}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          {title}
        </h3>
      )}

      <div className="flex items-center justify-center">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {slices.map((slice, index) => (
            <path
              key={index}
              d={getSlicePath(slice.startAngle, slice.endAngle, size / 2)}
              fill={slice.color}
              stroke="white"
              strokeWidth="2"
              className="hover:opacity-80 transition-opacity cursor-pointer"
            />
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        {slices.map((slice, index) => (
          <div key={index} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded"
              style={{ backgroundColor: slice.color }}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {slice.label} ({slice.percentage.toFixed(1)}%)
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

// Simple line chart using SVG
interface LineChartProps {
  data: { x: string; y: number }[];
  title?: string;
  width?: number;
  height?: number;
  className?: string;
}

export function LineChart({
  data,
  title,
  width = 600,
  height = 200,
  className = '',
}: LineChartProps) {
  const padding = 40;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const maxY = Math.max(...data.map((d) => d.y));
  const minY = Math.min(...data.map((d) => d.y));
  const rangeY = maxY - minY || 1;

  const getX = (index: number) => (index / (data.length - 1)) * chartWidth + padding;
  const getY = (value: number) => chartHeight - ((value - minY) / rangeY) * chartHeight + padding;

  // Create path
  const pathD = data.map((point, index) => {
    const x = getX(index);
    const y = getY(point.y);
    return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');

  return (
    <Card className={className}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          {title}
        </h3>
      )}

      <svg width={width} height={height} className="mx-auto">
        {/* Grid lines */}
        {[0, 1, 2, 3, 4].map((i) => {
          const y = (chartHeight / 4) * i + padding;
          return (
            <line
              key={i}
              x1={padding}
              y1={y}
              x2={width - padding}
              y2={y}
              stroke="currentColor"
              className="text-gray-200 dark:text-gray-700"
              strokeWidth="1"
            />
          );
        })}

        {/* Line */}
        <path
          d={pathD}
          fill="none"
          stroke="#3B82F6"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Data points */}
        {data.map((point, index) => (
          <circle
            key={index}
            cx={getX(index)}
            cy={getY(point.y)}
            r="4"
            fill="#3B82F6"
            className="hover:r-6 transition-all cursor-pointer"
          />
        ))}

        {/* X-axis labels */}
        {data.map((point, index) => (
          <text
            key={index}
            x={getX(index)}
            y={height - padding + 20}
            textAnchor="middle"
            className="text-xs fill-gray-600 dark:fill-gray-400"
          >
            {point.x}
          </text>
        ))}
      </svg>
    </Card>
  );
}
