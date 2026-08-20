/**
 * Monitoring status helpers.
 *
 * Pure derivations over the monitoring payloads: mapping a service or alert onto
 * a display tone, counting healthy services / firing alerts and detecting the
 * log level of a raw log line. Kept out of the component modules so the
 * presentational components only export components.
 */

import type {
  MonitoringAlert,
  MonitoringLogEntry,
  MonitoringService,
} from '@/types/openevolve';
import { formatMetricNumber, humanizeKey } from './formatters';

// ==================== Services ====================

export type ServiceTone = 'green' | 'yellow' | 'red' | 'gray';

const HEALTHY_STATUSES = ['healthy', 'ok', 'up', 'online', 'running', 'available'];
const DEGRADED_STATUSES = ['degraded', 'warning', 'warn', 'starting', 'pending'];
const FAILING_STATUSES = [
  'unhealthy',
  'down',
  'error',
  'failed',
  'offline',
  'unavailable',
];

/** Map a service payload onto a display tone. */
export function serviceTone(service: MonitoringService): ServiceTone {
  if (service.healthy === true) return 'green';
  if (service.healthy === false) return 'red';

  const status = (service.status ?? '').toLowerCase();
  if (HEALTHY_STATUSES.includes(status)) return 'green';
  if (DEGRADED_STATUSES.includes(status)) return 'yellow';
  if (FAILING_STATUSES.includes(status)) return 'red';
  return 'gray';
}

/** True when a service is reporting a healthy state. */
export function isServiceHealthy(service: MonitoringService): boolean {
  return serviceTone(service) === 'green';
}

/** Number of services currently reporting healthy. */
export function countHealthyServices(services: MonitoringService[]): number {
  return services.filter(isServiceHealthy).length;
}

/** Display label for a service status cell. */
export function serviceStatusLabel(service: MonitoringService): string {
  if (service.status && service.status.length > 0) return service.status;
  if (service.healthy === true) return 'healthy';
  if (service.healthy === false) return 'unhealthy';
  return 'unknown';
}

// ==================== Alerts ====================

/**
 * True when an alert is currently firing. The backend may report `triggered`
 * (evaluated state) and/or `active` (rule enabled / currently active), so
 * `triggered` wins whenever it is present.
 */
export function isAlertFiring(alert: MonitoringAlert): boolean {
  if (typeof alert.triggered === 'boolean') return alert.triggered;
  return alert.active === true;
}

/** Number of alerts currently firing. */
export function countFiringAlerts(alerts: MonitoringAlert[]): number {
  return alerts.filter(isAlertFiring).length;
}

/** Best-effort human readable title for an alert rule. */
export function alertTitle(alert: MonitoringAlert): string {
  if (alert.name && alert.name.length > 0) return humanizeKey(alert.name);
  if (alert.metric_name && alert.metric_name.length > 0) {
    return humanizeKey(alert.metric_name);
  }
  return 'Unnamed alert';
}

/** `cpu_percent > 90` style summary of the rule, when enough fields exist. */
export function alertConditionText(alert: MonitoringAlert): string | null {
  const parts: string[] = [];
  if (alert.metric_name) parts.push(alert.metric_name);
  if (alert.condition) parts.push(alert.condition);
  if (typeof alert.threshold === 'number') {
    parts.push(formatMetricNumber(alert.threshold));
  }
  return parts.length > 0 ? parts.join(' ') : null;
}

/** Firing alerts first, then alphabetically by title. */
export function sortAlerts(alerts: MonitoringAlert[]): MonitoringAlert[] {
  return [...alerts].sort((a, b) => {
    const firingDelta = Number(isAlertFiring(b)) - Number(isAlertFiring(a));
    if (firingDelta !== 0) return firingDelta;
    return alertTitle(a).localeCompare(alertTitle(b));
  });
}

// ==================== Logs ====================

export type LogTone = 'error' | 'warn' | 'info' | 'debug' | 'plain';

/** Detect a log level from the raw line so it can be colour-coded. */
export function logLineTone(line: string): LogTone {
  const upper = line.toUpperCase();
  if (
    upper.includes('ERROR') ||
    upper.includes('CRITICAL') ||
    upper.includes('FATAL') ||
    upper.includes('TRACEBACK') ||
    upper.includes('EXCEPTION')
  ) {
    return 'error';
  }
  if (upper.includes('WARN')) return 'warn';
  if (upper.includes('INFO')) return 'info';
  if (upper.includes('DEBUG') || upper.includes('TRACE')) return 'debug';
  return 'plain';
}

/** Distinct log sources present in a batch of entries. */
export function collectLogSources(entries: MonitoringLogEntry[]): string[] {
  return [...new Set(entries.map((entry) => entry.source))].sort();
}
