/**
 * MonitoringSummary
 * Top summary block for the monitoring view, built from
 * `getMonitoringDashboard()`:
 *  - health status + uptime tiles (with alert/service roll-ups from the
 *    sibling endpoints so the header reads as one snapshot)
 *  - system resource gauges (`system.system`, e.g. `cpu_percent`)
 *  - workflow counters (`workflow`)
 *  - the `recent_metrics` bag rendered as compact key/value chips
 */

import { Activity, AlertCircle, Clock, Cpu } from 'lucide-react';
import type { MonitoringDashboardMetrics } from '@/types/openevolve';
import { StatTile } from './StatTile';
import {
  EMPTY_VALUE,
  formatMetricValue,
  formatTimestamp,
  formatUnknownValue,
  formatUptime,
  humanizeKey,
  isPercentKey,
  toNumericEntries,
  toRecordEntries,
  utilizationTone,
} from './formatters';

const gaugeBarStyles: Record<'green' | 'yellow' | 'red', string> = {
  green: 'bg-green-500',
  yellow: 'bg-yellow-500',
  red: 'bg-red-500',
};

function GaugeRow({ label, value }: { label: string; value: number }) {
  const clamped = Math.min(Math.max(value, 0), 100);
  const tone = utilizationTone(clamped);
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-3 py-2 dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="truncate text-gray-600 dark:text-gray-400" title={label}>
          {label}
        </span>
        <span className="font-mono text-gray-900 dark:text-white">
          {clamped.toFixed(1)}%
        </span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          className={`h-full rounded-full ${gaugeBarStyles[tone]}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function CompactStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-3 py-2 dark:border-gray-700 dark:bg-gray-800">
      <p className="truncate text-xs text-gray-600 dark:text-gray-400" title={label}>
        {label}
      </p>
      <p className="mt-0.5 truncate text-lg font-semibold text-gray-900 dark:text-white">
        {value}
      </p>
    </div>
  );
}

interface MonitoringSummaryProps {
  dashboard: MonitoringDashboardMetrics | null;
  loading?: boolean;
  /** Firing alert count from `getMonitoringAlerts()`. */
  firingAlertCount?: number;
  /** Healthy service count from `getMonitoringServices()`. */
  healthyServiceCount?: number;
  /** Total monitored service count from `getMonitoringServices()`. */
  serviceCount?: number;
}

export function MonitoringSummary({
  dashboard,
  loading = false,
  firingAlertCount,
  healthyServiceCount,
  serviceCount,
}: MonitoringSummaryProps) {
  if (loading && !dashboard) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((index) => (
          <div
            key={index}
            className="h-28 animate-pulse rounded-lg border border-gray-200 bg-gray-100 dark:border-gray-700 dark:bg-gray-700/40"
          />
        ))}
      </div>
    );
  }

  if (!dashboard) {
    return (
      <p className="py-6 text-center text-sm text-gray-500 dark:text-gray-400">
        No dashboard snapshot available.
      </p>
    );
  }

  const health = dashboard.health;
  const systemEntries = toNumericEntries(dashboard.system?.system);
  const workflowEntries = toNumericEntries(dashboard.workflow);
  const recentEntries = toRecordEntries(dashboard.recent_metrics);

  const firing = firingAlertCount ?? 0;
  const healthy = healthyServiceCount ?? 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Backend health"
          value={health?.status ?? 'unknown'}
          tone={health?.healthy ? 'green' : 'red'}
          hint={health?.healthy ? 'All checks passing' : 'Checks reporting problems'}
          icon={<Activity className="h-5 w-5" />}
        />
        <StatTile
          label="Uptime"
          value={formatUptime(health?.uptime_seconds)}
          tone="blue"
          hint={`Snapshot ${formatTimestamp(dashboard.timestamp)}`}
          icon={<Clock className="h-5 w-5" />}
        />
        <StatTile
          label="Firing alerts"
          value={firingAlertCount === undefined ? EMPTY_VALUE : firing}
          tone={firing > 0 ? 'red' : 'green'}
          hint={firing > 0 ? 'Needs attention' : 'No active alerts'}
          icon={<AlertCircle className="h-5 w-5" />}
        />
        <StatTile
          label="Services healthy"
          value={
            serviceCount === undefined ? EMPTY_VALUE : `${healthy}/${serviceCount}`
          }
          tone={
            serviceCount === undefined
              ? 'gray'
              : healthy === serviceCount
                ? 'green'
                : 'yellow'
          }
          hint="From the service health sweep"
          icon={<Cpu className="h-5 w-5" />}
        />
      </div>

      {systemEntries.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
            System resources
          </h4>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {systemEntries.map(([key, value]) =>
              isPercentKey(key) ? (
                <GaugeRow key={key} label={humanizeKey(key)} value={value} />
              ) : (
                <CompactStat
                  key={key}
                  label={humanizeKey(key)}
                  value={formatMetricValue(key, value)}
                />
              )
            )}
          </div>
        </section>
      )}

      {workflowEntries.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
            Workflow activity
          </h4>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {workflowEntries.map(([key, value]) => (
              <CompactStat
                key={key}
                label={humanizeKey(key)}
                value={formatMetricValue(key, value)}
              />
            ))}
          </div>
        </section>
      )}

      {recentEntries.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
            Recent metrics
          </h4>
          <ul className="divide-y divide-gray-200 rounded-lg border border-gray-200 bg-white px-3 dark:divide-gray-700 dark:border-gray-700 dark:bg-gray-800">
            {recentEntries.map(([name, fields]) => (
              <li key={name} className="py-2">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {humanizeKey(name)}
                </p>
                <div className="mt-1 flex flex-wrap gap-1">
                  {Object.entries(fields).map(([key, value]) => (
                    <span
                      key={key}
                      className="inline-flex items-center rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-700 dark:bg-gray-700 dark:text-gray-200"
                      title={`${key}: ${formatUnknownValue(value)}`}
                    >
                      {humanizeKey(key)}: {formatUnknownValue(value)}
                    </span>
                  ))}
                  {Object.keys(fields).length === 0 && (
                    <span className="text-xs text-gray-400 dark:text-gray-500">
                      {EMPTY_VALUE}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
