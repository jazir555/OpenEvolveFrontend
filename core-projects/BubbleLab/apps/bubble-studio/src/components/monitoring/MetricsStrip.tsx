/**
 * MetricsStrip
 * Horizontally scrollable strip of the raw metric samples returned by
 * `getMonitoringMetrics()`. Values are formatted from the metric name
 * (percent / bytes / seconds / ms) and labels are rendered as chips.
 */

import type { MonitoringMetric } from '@/types/openevolve';
import {
  EMPTY_VALUE,
  formatMetricValue,
  formatSince,
  formatUnknownValue,
  humanizeKey,
} from './formatters';

interface MetricsStripProps {
  metrics: MonitoringMetric[];
  loading?: boolean;
  /** Maximum number of tiles rendered (keeps the strip readable). */
  maxItems?: number;
}

export function MetricsStrip({
  metrics,
  loading = false,
  maxItems = 24,
}: MetricsStripProps) {
  if (loading && metrics.length === 0) {
    return (
      <div className="flex gap-3 overflow-hidden">
        {[0, 1, 2, 3].map((index) => (
          <div
            key={index}
            className="h-24 w-44 shrink-0 animate-pulse rounded-lg border border-gray-200 bg-gray-100 dark:border-gray-700 dark:bg-gray-700/40"
          />
        ))}
      </div>
    );
  }

  if (metrics.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-gray-500 dark:text-gray-400">
        No metric samples in the selected window.
      </p>
    );
  }

  const visible = metrics.slice(0, maxItems);

  return (
    <div className="space-y-2">
      <div className="thin-scrollbar flex gap-3 overflow-x-auto pb-2">
        {visible.map((metric, index) => {
          const name = metric.name ?? 'metric';
          const labels = Object.entries(metric.labels ?? {});
          return (
            <div
              key={`${name}-${index}`}
              className="w-48 shrink-0 rounded-lg border border-gray-200 bg-white p-3 shadow-sm dark:border-gray-700 dark:bg-gray-800"
            >
              <p
                className="truncate text-xs font-medium text-gray-600 dark:text-gray-400"
                title={metric.description ?? name}
              >
                {humanizeKey(name)}
              </p>
              <p className="mt-1 truncate text-xl font-semibold text-gray-900 dark:text-white">
                {typeof metric.value === 'number'
                  ? formatMetricValue(name, metric.value)
                  : EMPTY_VALUE}
              </p>

              <div className="mt-2 flex flex-wrap gap-1">
                {metric.type && (
                  <span className="inline-flex items-center rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-700 dark:bg-gray-700 dark:text-gray-200">
                    {metric.type}
                  </span>
                )}
                {labels.slice(0, 2).map(([key, value]) => (
                  <span
                    key={key}
                    className="inline-flex max-w-full items-center truncate rounded bg-blue-50 px-1.5 py-0.5 text-xs text-blue-700 dark:bg-blue-900/20 dark:text-blue-400"
                    title={`${key}: ${formatUnknownValue(value)}`}
                  >
                    {key}: {formatUnknownValue(value)}
                  </span>
                ))}
              </div>

              <p className="mt-2 truncate text-xs text-gray-400 dark:text-gray-500">
                {formatSince(metric.timestamp)}
              </p>
            </div>
          );
        })}
      </div>

      {metrics.length > visible.length && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Showing {visible.length} of {metrics.length} samples.
        </p>
      )}
    </div>
  );
}
