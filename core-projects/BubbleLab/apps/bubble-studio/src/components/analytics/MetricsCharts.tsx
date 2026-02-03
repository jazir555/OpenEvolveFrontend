/**
 * MetricsCharts Component
 * Display analytics charts and graphs
 */

import { Card } from '../common/Card';

interface MetricData {
  name: string;
  value: number;
  change?: number;
}

export function MetricsCharts() {
  const executionData: MetricData[] = [
    { name: 'Mon', value: 12 },
    { name: 'Tue', value: 19 },
    { name: 'Wed', value: 15 },
    { name: 'Thu', value: 25 },
    { name: 'Fri', value: 22 },
    { name: 'Sat', value: 10 },
    { name: 'Sun', value: 8 },
  ];

  const successRateData = [
    { name: 'Week 1', value: 75 },
    { name: 'Week 2', value: 82 },
    { name: 'Week 3', value: 78 },
    { name: 'Week 4', value: 88 },
  ];

  const maxValue = Math.max(...executionData.map((d) => d.value));

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Execution Timeline Chart */}
      <Card title="Weekly Executions" subtitle="Workflow executions over time">
        <div className="space-y-4">
          {executionData.map((data, index) => (
            <div key={index} className="flex items-center gap-4">
              <div className="w-12 text-sm text-gray-600 dark:text-gray-400">
                {data.name}
              </div>
              <div className="flex-1">
                <div className="h-8 rounded bg-gray-200 dark:bg-gray-700 relative">
                  <div
                    className="h-full rounded bg-blue-600 transition-all"
                    style={{ width: `${(data.value / maxValue) * 100}%` }}
                  />
                  <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white mix-blend-multiply">
                    {data.value}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Success Rate Trend */}
      <Card title="Success Rate Trend" subtitle="4-week overview">
        <div className="space-y-4">
          {successRateData.map((data, index) => (
            <div key={index} className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">{data.name}</span>
              <div className="flex items-center gap-2">
                <div className="w-32 h-2 rounded-full bg-gray-200 dark:bg-gray-700">
                  <div
                    className={`h-2 rounded-full ${
                      data.value >= 80
                        ? 'bg-green-500'
                        : data.value >= 70
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${data.value}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-white w-12 text-right">
                  {data.value}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Performance Metrics */}
      <Card title="Performance Metrics" subtitle="Average execution times">
        <div className="grid grid-cols-2 gap-4">
          <MetricCard label="Fastest" value="12.5s" improvement="↓ 15%" />
          <MetricCard label="Slowest" value="85.3s" improvement="↑ 5%" />
          <MetricCard label="Average" value="42.7s" improvement="↓ 8%" />
          <MetricCard label="Median" value="38.2s" improvement="↓ 10%" />
        </div>
      </Card>

      {/* Token Usage */}
      <Card title="Token Usage" subtitle="Tokens consumed per workflow type">
        <div className="space-y-4">
          <TokenBar label="Text Generation" used={45000} total={50000} color="blue" />
          <TokenBar label="Code Generation" used={32000} total={50000} color="green" />
          <TokenBar label="Image Analysis" used={28000} total={50000} color="purple" />
          <TokenBar label="Multi-modal" used={18000} total={50000} color="yellow" />
        </div>
      </Card>
    </div>
  );
}

function MetricCard({ label, value, improvement }: { label: string; value: string; improvement: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 text-center dark:border-gray-700 dark:bg-gray-800">
      <p className="text-sm text-gray-600 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-xl font-semibold text-gray-900 dark:text-white">{value}</p>
      <p className="mt-1 text-xs text-green-600 dark:text-green-400">{improvement}</p>
    </div>
  );
}

function TokenBar({ label, used, total, color }: { label: string; used: number; total: number; color: string }) {
  const percentage = (used / total) * 100;
  const colorStyles = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    purple: 'bg-purple-600',
    yellow: 'bg-yellow-600',
  };

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="font-medium text-gray-700 dark:text-gray-300">{label}</span>
        <span className="text-gray-900 dark:text-white">
          {used.toLocaleString()} / {total.toLocaleString()}
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
        <div className={`h-2 rounded-full ${colorStyles[color as keyof typeof colorStyles]}`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}
