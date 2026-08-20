/**
 * Monitoring component barrel.
 *
 * Re-exports the public surface of the real-time monitoring view so consumers
 * (routing / `App.tsx`) can import from a single module, e.g.:
 *   import { MonitoringView } from '@/components/monitoring';
 */

export { MonitoringView, type MonitoringViewProps } from './MonitoringView';
export { MonitoringSummary } from './MonitoringSummary';
export { ServiceHealthGrid } from './ServiceHealthGrid';
export { AlertList } from './AlertList';
export { MetricsStrip } from './MetricsStrip';
export { LogViewer } from './LogViewer';
export { StatTile, type StatTone } from './StatTile';

export {
  useMonitoringData,
  type MonitoringData,
  type MonitoringErrors,
  type MonitoringQueryOptions,
  type MonitoringSectionId,
  type UseMonitoringDataResult,
} from './useMonitoringData';

export {
  alertConditionText,
  alertTitle,
  collectLogSources,
  countFiringAlerts,
  countHealthyServices,
  isAlertFiring,
  isServiceHealthy,
  logLineTone,
  serviceStatusLabel,
  serviceTone,
  sortAlerts,
  type LogTone,
  type ServiceTone,
} from './status';

export {
  EMPTY_VALUE,
  formatExecutionTime,
  formatMetricNumber,
  formatMetricValue,
  formatSince,
  formatTimestamp,
  formatUnknownValue,
  formatUptime,
  humanizeKey,
  isPercentKey,
  toNumericEntries,
  toRecordEntries,
  utilizationTone,
} from './formatters';
