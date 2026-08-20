/**
 * MonitoringView
 * Real-time monitoring dashboard for the OpenEvolve backend.
 *
 * Composes one snapshot from the monitoring endpoints exposed by
 * `openevolveApi` (see `useMonitoringData`):
 *  - `getMonitoringDashboard()` -> summary tiles, system gauges, workflow counters
 *  - `getMonitoringServices()`  -> service health grid
 *  - `getMonitoringAlerts()`    -> alert rules / firing state
 *  - `getMonitoringMetrics()`   -> metrics strip (name + time-window filters)
 *  - `getMonitoringLogs()`      -> recent log lines (source + limit filters)
 *
 * The header exposes a manual "Refresh" plus an optional auto-refresh interval
 * (a `setInterval` poll owned by `useMonitoringData`, cleared on unmount).
 * Every section renders its own loading, empty and error state so a single
 * failing endpoint never blanks the whole dashboard.
 */

import { useState, type FormEvent, type ReactNode } from 'react';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Cpu,
  FileText,
  RefreshCw,
} from 'lucide-react';
import { Alert } from '../common/Alert';
import { Badge, type BadgeVariant } from '../common/Badge';
import { Card } from '../common/Card';
import { AlertList } from './AlertList';
import { LogViewer } from './LogViewer';
import { MetricsStrip } from './MetricsStrip';
import { MonitoringSummary } from './MonitoringSummary';
import { ServiceHealthGrid } from './ServiceHealthGrid';
import { countFiringAlerts, countHealthyServices, isAlertFiring } from './status';
import { useMonitoringData } from './useMonitoringData';

type TabId = 'overview' | 'services' | 'alerts' | 'logs';

const TABS: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'services', label: 'Services' },
  { id: 'alerts', label: 'Alerts' },
  { id: 'logs', label: 'Logs' },
];

const AUTO_REFRESH_OPTIONS: { label: string; value: number }[] = [
  { label: 'Off', value: 0 },
  { label: '10s', value: 10_000 },
  { label: '30s', value: 30_000 },
  { label: '60s', value: 60_000 },
];

const METRIC_WINDOW_OPTIONS: { label: string; value: number }[] = [
  { label: 'Last 15 min', value: 15 },
  { label: 'Last hour', value: 60 },
  { label: 'Last 24 hours', value: 1440 },
  { label: 'All time', value: 0 },
];

const LOG_LIMIT_OPTIONS = [50, 100, 200, 500];

const controlClassName =
  'rounded-md border border-gray-300 bg-white px-2 py-1 text-xs text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white';

interface MonitoringSectionProps {
  title: string;
  subtitle?: string;
  icon: ReactNode;
  error?: string;
  actions?: ReactNode;
  children: ReactNode;
}

function MonitoringSection({
  title,
  subtitle,
  icon,
  error,
  actions,
  children,
}: MonitoringSectionProps) {
  return (
    <Card
      title={title}
      subtitle={subtitle}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          {actions}
          <span className="text-gray-400 dark:text-gray-500">{icon}</span>
        </div>
      }
    >
      {error && (
        <div className="mb-4">
          <Alert type="error" title={`Failed to load ${title.toLowerCase()}`}>
            {error}
          </Alert>
        </div>
      )}
      {children}
    </Card>
  );
}

export interface MonitoringViewProps {
  /** Initial auto-refresh interval in ms (`0` starts with polling disabled). */
  defaultAutoRefreshMs?: number;
  className?: string;
}

export function MonitoringView({
  defaultAutoRefreshMs = 30_000,
  className = '',
}: MonitoringViewProps) {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [autoRefreshMs, setAutoRefreshMs] = useState(defaultAutoRefreshMs);

  const [metricNameDraft, setMetricNameDraft] = useState('');
  const [metricName, setMetricName] = useState('');
  const [metricWindowMinutes, setMetricWindowMinutes] = useState(60);

  const [logSourceDraft, setLogSourceDraft] = useState('');
  const [logSource, setLogSource] = useState('');
  const [logLimit, setLogLimit] = useState(200);

  const {
    dashboard,
    services,
    servicesTimestamp,
    alerts,
    metrics,
    logEntries,
    logTotal,
    errors,
    isInitialLoading,
    isRefreshing,
    lastUpdated,
    refresh,
  } = useMonitoringData({
    logLimit,
    logSource: logSource.trim().length > 0 ? logSource.trim() : undefined,
    metricName: metricName.trim().length > 0 ? metricName.trim() : undefined,
    metricWindowMinutes,
    autoRefreshMs,
  });

  const firingAlertCount = countFiringAlerts(alerts);
  const healthyServiceCount = countHealthyServices(services);
  const firingAlerts = alerts.filter(isAlertFiring);
  const failedSectionCount = Object.keys(errors).length;

  const health = dashboard?.health;
  const healthVariant: BadgeVariant = health
    ? health.healthy
      ? 'green'
      : 'red'
    : 'gray';
  const healthLabel = health ? health.status : 'unknown';

  const tabCounts: Record<TabId, number | null> = {
    overview: null,
    services: services.length,
    alerts: firingAlertCount > 0 ? firingAlertCount : alerts.length,
    logs: logEntries.length,
  };

  const handleMetricFilterSubmit = (event: FormEvent) => {
    event.preventDefault();
    setMetricName(metricNameDraft);
  };

  const handleLogFilterSubmit = (event: FormEvent) => {
    event.preventDefault();
    setLogSource(logSourceDraft);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* ============================ Header ============================ */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Monitoring
            </h1>
            <Badge variant={healthVariant}>{healthLabel}</Badge>
          </div>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Live OpenEvolve backend health, metrics, alerts and logs.
          </p>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {lastUpdated
              ? `Updated ${lastUpdated.toLocaleTimeString()}`
              : 'Waiting for the first snapshot…'}
            {autoRefreshMs > 0 && ` · auto-refresh every ${Math.round(autoRefreshMs / 1000)}s`}
            {failedSectionCount > 0 &&
              ` · ${failedSectionCount} section${failedSectionCount === 1 ? '' : 's'} failed to load`}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
            Auto-refresh
            <select
              value={autoRefreshMs}
              onChange={(event) => setAutoRefreshMs(Number(event.target.value))}
              className={controlClassName}
            >
              {AUTO_REFRESH_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={refresh}
            disabled={isRefreshing}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            {isRefreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* ============================= Tabs ============================= */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8" aria-label="Monitoring tabs">
          {TABS.map((tab) => {
            const count = tabCounts[tab.id];
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 whitespace-nowrap border-b-2 px-1 py-3 text-sm font-medium transition-colors ${
                  isActive
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                {tab.label}
                {count !== null && count > 0 && (
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* =========================== Overview =========================== */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <MonitoringSection
            title="System Overview"
            subtitle="Health, uptime and resource snapshot"
            icon={<Activity className="h-5 w-5" />}
            error={errors.dashboard}
          >
            <MonitoringSummary
              dashboard={dashboard}
              loading={isInitialLoading}
              firingAlertCount={errors.alerts ? undefined : firingAlertCount}
              healthyServiceCount={errors.services ? undefined : healthyServiceCount}
              serviceCount={errors.services ? undefined : services.length}
            />
          </MonitoringSection>

          <MonitoringSection
            title="Metrics"
            subtitle={`${metrics.length} sample${metrics.length === 1 ? '' : 's'}${
              metricName.trim().length > 0 ? ` for "${metricName.trim()}"` : ''
            }`}
            icon={<BarChart3 className="h-5 w-5" />}
            error={errors.metrics}
            actions={
              <form
                onSubmit={handleMetricFilterSubmit}
                className="flex flex-wrap items-center gap-2"
              >
                <input
                  type="text"
                  value={metricNameDraft}
                  onChange={(event) => setMetricNameDraft(event.target.value)}
                  placeholder="Metric name…"
                  className={controlClassName}
                />
                <select
                  value={metricWindowMinutes}
                  onChange={(event) =>
                    setMetricWindowMinutes(Number(event.target.value))
                  }
                  className={controlClassName}
                >
                  {METRIC_WINDOW_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <button
                  type="submit"
                  className="rounded-md border border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800"
                >
                  Apply
                </button>
              </form>
            }
          >
            <MetricsStrip metrics={metrics} loading={isInitialLoading} />
          </MonitoringSection>

          <MonitoringSection
            title="Firing Alerts"
            subtitle={
              firingAlerts.length > 0
                ? `${firingAlerts.length} alert${
                    firingAlerts.length === 1 ? ' needs' : 's need'
                  } attention`
                : 'No alerts are firing'
            }
            icon={<AlertCircle className="h-5 w-5" />}
            error={errors.alerts}
          >
            {firingAlerts.length > 0 ? (
              <AlertList alerts={firingAlerts} loading={isInitialLoading} />
            ) : (
              <p className="py-6 text-center text-sm text-gray-500 dark:text-gray-400">
                {isInitialLoading
                  ? 'Loading alerts…'
                  : 'All configured alert rules are within their thresholds.'}
              </p>
            )}
          </MonitoringSection>
        </div>
      )}

      {/* =========================== Services =========================== */}
      {activeTab === 'services' && (
        <MonitoringSection
          title="Service Health"
          subtitle={`${healthyServiceCount}/${services.length} healthy`}
          icon={<Cpu className="h-5 w-5" />}
          error={errors.services}
        >
          <ServiceHealthGrid
            services={services}
            timestamp={servicesTimestamp}
            loading={isInitialLoading}
          />
        </MonitoringSection>
      )}

      {/* ============================ Alerts ============================ */}
      {activeTab === 'alerts' && (
        <MonitoringSection
          title="Alerts"
          subtitle={`${firingAlertCount} firing of ${alerts.length} rule${
            alerts.length === 1 ? '' : 's'
          }`}
          icon={<AlertCircle className="h-5 w-5" />}
          error={errors.alerts}
        >
          <AlertList alerts={alerts} loading={isInitialLoading} />
        </MonitoringSection>
      )}

      {/* ============================= Logs ============================= */}
      {activeTab === 'logs' && (
        <MonitoringSection
          title="Recent Logs"
          subtitle={
            logSource.trim().length > 0
              ? `Source "${logSource.trim()}"`
              : 'All sources'
          }
          icon={<FileText className="h-5 w-5" />}
          error={errors.logs}
          actions={
            <form
              onSubmit={handleLogFilterSubmit}
              className="flex flex-wrap items-center gap-2"
            >
              <input
                type="text"
                value={logSourceDraft}
                onChange={(event) => setLogSourceDraft(event.target.value)}
                placeholder="Source…"
                className={controlClassName}
              />
              <select
                value={logLimit}
                onChange={(event) => setLogLimit(Number(event.target.value))}
                className={controlClassName}
              >
                {LOG_LIMIT_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option} lines
                  </option>
                ))}
              </select>
              <button
                type="submit"
                className="rounded-md border border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800"
              >
                Apply
              </button>
            </form>
          }
        >
          <LogViewer
            entries={logEntries}
            total={logTotal}
            loading={isInitialLoading}
          />
        </MonitoringSection>
      )}
    </div>
  );
}
