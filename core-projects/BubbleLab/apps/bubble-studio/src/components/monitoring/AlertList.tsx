/**
 * AlertList
 * Renders the alert rules returned by `getMonitoringAlerts()`. Firing alerts are
 * sorted to the top and highlighted; each row shows the rule condition
 * (`metric condition threshold`) and the latest observed value.
 */

import type { MonitoringAlert } from '@/types/openevolve';
import { Badge } from '../common/Badge';
import { EMPTY_VALUE, formatMetricNumber } from './formatters';
import { alertConditionText, alertTitle, isAlertFiring, sortAlerts } from './status';

interface AlertListProps {
  alerts: MonitoringAlert[];
  loading?: boolean;
}

export function AlertList({ alerts, loading = false }: AlertListProps) {
  if (loading && alerts.length === 0) {
    return (
      <div className="space-y-2">
        {[0, 1, 2].map((index) => (
          <div
            key={index}
            className="h-16 animate-pulse rounded-lg border border-gray-200 bg-gray-100 dark:border-gray-700 dark:bg-gray-700/40"
          />
        ))}
      </div>
    );
  }

  if (alerts.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-gray-500 dark:text-gray-400">
        No alert rules configured.
      </p>
    );
  }

  // Firing alerts first, then alphabetically by title.
  const ordered = sortAlerts(alerts);

  return (
    <ul className="space-y-2">
      {ordered.map((alert, index) => {
        const firing = isAlertFiring(alert);
        const condition = alertConditionText(alert);
        return (
          <li
            key={`${alert.name ?? alert.metric_name ?? 'alert'}-${index}`}
            className={`rounded-lg border p-3 ${
              firing
                ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
                : 'border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800'
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium text-gray-900 dark:text-white">
                    {alertTitle(alert)}
                  </span>
                  <Badge variant={firing ? 'red' : 'green'} size="sm">
                    {firing ? 'firing' : 'ok'}
                  </Badge>
                  {alert.active === false && (
                    <Badge variant="gray" size="sm">
                      disabled
                    </Badge>
                  )}
                </div>
                {alert.description && (
                  <p className="mt-1 text-xs text-gray-600 dark:text-gray-400">
                    {alert.description}
                  </p>
                )}
                {condition && (
                  <p className="mt-1 font-mono text-xs text-gray-500 dark:text-gray-400">
                    {condition}
                  </p>
                )}
              </div>

              <div className="shrink-0 text-right">
                <p
                  className={`text-sm font-semibold ${
                    firing
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-gray-900 dark:text-white'
                  }`}
                >
                  {typeof alert.latest_value === 'number'
                    ? formatMetricNumber(alert.latest_value)
                    : EMPTY_VALUE}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">latest</p>
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
