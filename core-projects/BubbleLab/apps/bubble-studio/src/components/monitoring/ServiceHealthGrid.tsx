/**
 * ServiceHealthGrid
 * Grid of per-service health cards built from `getMonitoringServices()`.
 * Each card shows the reported status, the check response time and the last
 * check timestamp, plus the raw error string when a check failed.
 */

import type { MonitoringService } from '@/types/openevolve';
import { Badge, type BadgeVariant } from '../common/Badge';
import { formatExecutionTime, formatSince } from './formatters';
import { serviceStatusLabel, serviceTone, type ServiceTone } from './status';

const dotStyles: Record<ServiceTone, string> = {
  green: 'bg-green-500',
  yellow: 'bg-yellow-500',
  red: 'bg-red-500',
  gray: 'bg-gray-400',
};

const badgeVariants: Record<ServiceTone, BadgeVariant> = {
  green: 'green',
  yellow: 'yellow',
  red: 'red',
  gray: 'gray',
};

interface ServiceHealthGridProps {
  services: MonitoringService[];
  /** Timestamp of the health sweep, when the backend reports one. */
  timestamp?: string | null;
  loading?: boolean;
}

export function ServiceHealthGrid({
  services,
  timestamp,
  loading = false,
}: ServiceHealthGridProps) {
  if (loading && services.length === 0) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {[0, 1, 2].map((index) => (
          <div
            key={index}
            className="h-28 animate-pulse rounded-lg border border-gray-200 bg-gray-100 dark:border-gray-700 dark:bg-gray-700/40"
          />
        ))}
      </div>
    );
  }

  if (services.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-gray-500 dark:text-gray-400">
        No monitored services reported.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {services.map((service, index) => {
          const tone = serviceTone(service);
          return (
            <div
              key={`${service.name}-${index}`}
              className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className={`h-2.5 w-2.5 shrink-0 rounded-full ${dotStyles[tone]}`}
                    aria-hidden="true"
                  />
                  <span
                    className="truncate text-sm font-medium text-gray-900 dark:text-white"
                    title={service.name}
                  >
                    {service.name}
                  </span>
                </div>
                <Badge variant={badgeVariants[tone]} size="sm">
                  {serviceStatusLabel(service)}
                </Badge>
              </div>

              <dl className="mt-3 space-y-1 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center justify-between gap-2">
                  <dt>Response</dt>
                  <dd className="font-mono text-gray-700 dark:text-gray-300">
                    {formatExecutionTime(service.execution_time)}
                  </dd>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <dt>Checked</dt>
                  <dd className="text-gray-700 dark:text-gray-300">
                    {formatSince(service.timestamp)}
                  </dd>
                </div>
              </dl>

              {service.error && (
                <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-2 py-1 font-mono text-xs text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
                  {service.error}
                </p>
              )}
            </div>
          );
        })}
      </div>

      {timestamp && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Health sweep {formatSince(timestamp)}
        </p>
      )}
    </div>
  );
}
