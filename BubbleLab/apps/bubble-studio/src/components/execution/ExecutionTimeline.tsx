/**
 * ExecutionTimeline Component
 * Timeline view of workflow execution steps
 */

import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { TimeAgo } from '../common/TimeAgo';

interface TimelineEvent {
  id: string;
  timestamp: Date;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  description?: string;
  duration?: number;
}

interface ExecutionTimelineProps {
  events: TimelineEvent[];
  className?: string;
}

export function ExecutionTimeline({ events, className = '' }: ExecutionTimelineProps) {
  const typeIcons = {
    info: 'ℹ️',
    success: '✅',
    warning: '⚠️',
    error: '❌',
  };

  const typeColors = {
    info: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20',
    success: 'border-green-500 bg-green-50 dark:bg-green-900/20',
    warning: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
    error: 'border-red-500 bg-red-50 dark:bg-red-900/20',
  };

  return (
    <Card className={className}>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Execution Timeline
      </h3>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700" />

        {/* Events */}
        <div className="space-y-4">
          {events.map((event, index) => (
            <div key={event.id} className="relative pl-10">
              {/* Icon */}
              <div
                className={`absolute left-2.5 w-3 h-3 rounded-full border-2 ${typeColors[event.type]}`}
              >
                <span className="absolute -top-0.5 left-0.5 text-xs">
                  {typeIcons[event.type]}
                </span>
              </div>

              {/* Content */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                <div className="flex items-start justify-between mb-1">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                      {event.title}
                    </h4>
                    {event.description && (
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {event.description}
                      </p>
                    )}
                  </div>

                  <div className="flex flex-col items-end gap-1">
                    <TimeAgo date={event.timestamp} className="text-xs text-gray-500" />
                    {event.duration && (
                      <span className="text-xs text-gray-500">
                        {formatDuration(event.duration)}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}
