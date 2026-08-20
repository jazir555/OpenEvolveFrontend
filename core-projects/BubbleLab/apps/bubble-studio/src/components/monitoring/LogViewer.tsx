/**
 * LogViewer
 * Monospace panel for the log lines returned by `getMonitoringLogs()`.
 * Each entry keeps its `source` chip and the line is tinted by the log level
 * detected in the text (ERROR / WARN / INFO / DEBUG).
 */

import type { MonitoringLogEntry } from '@/types/openevolve';
import { logLineTone, type LogTone } from './status';

const toneStyles: Record<LogTone, string> = {
  error: 'text-red-600 dark:text-red-400',
  warn: 'text-yellow-600 dark:text-yellow-400',
  info: 'text-gray-700 dark:text-gray-300',
  debug: 'text-gray-500 dark:text-gray-500',
  plain: 'text-gray-700 dark:text-gray-300',
};

interface LogViewerProps {
  entries: MonitoringLogEntry[];
  /** Total lines available on the backend (may exceed the requested limit). */
  total: number;
  loading?: boolean;
  /** Tailwind max-height class for the scroll area. */
  maxHeightClassName?: string;
}

export function LogViewer({
  entries,
  total,
  loading = false,
  maxHeightClassName = 'max-h-[28rem]',
}: LogViewerProps) {
  if (loading && entries.length === 0) {
    return (
      <div className="h-48 animate-pulse rounded-lg border border-gray-200 bg-gray-100 dark:border-gray-700 dark:bg-gray-700/40" />
    );
  }

  if (entries.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-gray-500 dark:text-gray-400">
        No log lines available.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-500 dark:text-gray-400">
        Showing {entries.length} of {total} line{total === 1 ? '' : 's'}.
      </p>
      <div
        className={`thin-scrollbar ${maxHeightClassName} overflow-auto rounded-lg border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900/60`}
      >
        <ul className="divide-y divide-gray-200 dark:divide-gray-800">
          {entries.map((entry, index) => (
            <li
              key={`${entry.source}-${index}`}
              className="flex items-start gap-3 px-3 py-1.5"
            >
              <span
                className="shrink-0 rounded bg-gray-200 px-1.5 py-0.5 font-mono text-xs text-gray-700 dark:bg-gray-700 dark:text-gray-300"
                title={entry.source}
              >
                {entry.source}
              </span>
              <span
                className={`whitespace-pre-wrap break-all font-mono text-xs ${
                  toneStyles[logLineTone(entry.line)]
                }`}
              >
                {entry.line}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
