/**
 * useMonitoringData
 *
 * Fetches a full monitoring snapshot from the OpenEvolve monitoring endpoints
 * through `openevolveApi`:
 *  - `getMonitoringDashboard` -> system / health / workflow summary
 *  - `getMonitoringServices`  -> per-service health checks
 *  - `getMonitoringAlerts`    -> configured alert rules and their state
 *  - `getMonitoringMetrics`   -> raw metric samples for the metrics strip
 *  - `getMonitoringLogs`      -> recent log lines
 *
 * All five requests are issued in parallel with `Promise.allSettled` so a single
 * failing endpoint only degrades its own section instead of blanking the view.
 * The previous successful payload for a failing section is kept on screen and
 * annotated with an error message.
 *
 * A manual `refresh()` is always available; passing `autoRefreshMs` starts a
 * lightweight `setInterval` poll that is cleared on unmount / option change.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { openevolveApi } from '@/services/openevolveApi';
import type {
  MonitoringAlert,
  MonitoringDashboardMetrics,
  MonitoringLogEntry,
  MonitoringMetric,
  MonitoringService,
} from '@/types/openevolve';

/** Identifies each independently-fetched monitoring section. */
export type MonitoringSectionId =
  | 'dashboard'
  | 'services'
  | 'alerts'
  | 'metrics'
  | 'logs';

/** Per-section error messages (absent key == section loaded fine). */
export type MonitoringErrors = Partial<Record<MonitoringSectionId, string>>;

export interface MonitoringQueryOptions {
  /** Number of log lines to request (`getMonitoringLogs` limit). */
  logLimit?: number;
  /** Optional log source filter (e.g. `api`, `worker`). */
  logSource?: string;
  /** Optional metric name filter for `getMonitoringMetrics`. */
  metricName?: string;
  /**
   * Metric lookback window in minutes. When set, `start_time` is computed at
   * fetch time so the window always ends "now".
   */
  metricWindowMinutes?: number;
  /** Auto-refresh interval in ms. `0`/omitted disables auto-refresh. */
  autoRefreshMs?: number;
}

export interface MonitoringData {
  dashboard: MonitoringDashboardMetrics | null;
  services: MonitoringService[];
  servicesTimestamp: string | null;
  alerts: MonitoringAlert[];
  metrics: MonitoringMetric[];
  logEntries: MonitoringLogEntry[];
  logTotal: number;
}

export interface UseMonitoringDataResult extends MonitoringData {
  errors: MonitoringErrors;
  /** True until the first snapshot (successful or not) has been applied. */
  isInitialLoading: boolean;
  /** True while any fetch (initial, manual or automatic) is in flight. */
  isRefreshing: boolean;
  /** Timestamp of the last applied snapshot. */
  lastUpdated: Date | null;
  /** Trigger a manual refresh of every section. */
  refresh: () => void;
}

const EMPTY_DATA: MonitoringData = {
  dashboard: null,
  services: [],
  servicesTimestamp: null,
  alerts: [],
  metrics: [],
  logEntries: [],
  logTotal: 0,
};

function toMessage(reason: unknown, fallback: string): string {
  if (reason instanceof Error && reason.message.length > 0) return reason.message;
  if (typeof reason === 'string' && reason.length > 0) return reason;
  return fallback;
}

export function useMonitoringData(
  options: MonitoringQueryOptions = {}
): UseMonitoringDataResult {
  const {
    logLimit = 200,
    logSource,
    metricName,
    metricWindowMinutes,
    autoRefreshMs = 0,
  } = options;

  const [data, setData] = useState<MonitoringData>(EMPTY_DATA);
  const [errors, setErrors] = useState<MonitoringErrors>({});
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const isMountedRef = useRef(true);
  const isFetchingRef = useRef(false);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const load = useCallback(async () => {
    // Skip overlapping polls: a slow backend should not queue up requests.
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    setIsRefreshing(true);

    const startTime =
      metricWindowMinutes && metricWindowMinutes > 0
        ? new Date(Date.now() - metricWindowMinutes * 60_000).toISOString()
        : undefined;

    const [dashboardResult, servicesResult, alertsResult, metricsResult, logsResult] =
      await Promise.allSettled([
        openevolveApi.getMonitoringDashboard(),
        openevolveApi.getMonitoringServices(),
        openevolveApi.getMonitoringAlerts(),
        openevolveApi.getMonitoringMetrics({
          name: metricName,
          start_time: startTime,
        }),
        openevolveApi.getMonitoringLogs(logLimit, logSource),
      ]);

    isFetchingRef.current = false;
    if (!isMountedRef.current) return;

    const nextErrors: MonitoringErrors = {};
    const patch: Partial<MonitoringData> = {};

    if (dashboardResult.status === 'fulfilled') {
      patch.dashboard = dashboardResult.value;
    } else {
      nextErrors.dashboard = toMessage(
        dashboardResult.reason,
        'Failed to load the monitoring dashboard.'
      );
    }

    if (servicesResult.status === 'fulfilled') {
      patch.services = servicesResult.value.services ?? [];
      patch.servicesTimestamp = servicesResult.value.timestamp ?? null;
    } else {
      nextErrors.services = toMessage(
        servicesResult.reason,
        'Failed to load service health.'
      );
    }

    if (alertsResult.status === 'fulfilled') {
      patch.alerts = alertsResult.value.alerts ?? [];
    } else {
      nextErrors.alerts = toMessage(alertsResult.reason, 'Failed to load alerts.');
    }

    if (metricsResult.status === 'fulfilled') {
      patch.metrics = metricsResult.value.metrics ?? [];
    } else {
      nextErrors.metrics = toMessage(metricsResult.reason, 'Failed to load metrics.');
    }

    if (logsResult.status === 'fulfilled') {
      patch.logEntries = logsResult.value.entries ?? [];
      patch.logTotal = logsResult.value.total ?? 0;
    } else {
      nextErrors.logs = toMessage(logsResult.reason, 'Failed to load logs.');
    }

    setData((previous) => ({ ...previous, ...patch }));
    setErrors(nextErrors);
    setLastUpdated(new Date());
    setIsInitialLoading(false);
    setIsRefreshing(false);
  }, [logLimit, logSource, metricName, metricWindowMinutes]);

  const refresh = useCallback(() => {
    void load();
  }, [load]);

  // Initial load, plus a reload whenever the query options change.
  useEffect(() => {
    void load();
  }, [load]);

  // Lightweight auto-refresh poll with cleanup.
  useEffect(() => {
    if (!autoRefreshMs || autoRefreshMs <= 0) return;
    const timer = window.setInterval(() => {
      void load();
    }, autoRefreshMs);
    return () => {
      window.clearInterval(timer);
    };
  }, [autoRefreshMs, load]);

  return {
    ...data,
    errors,
    isInitialLoading,
    isRefreshing,
    lastUpdated,
    refresh,
  };
}
