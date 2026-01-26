/**
 * Execution Logs Component
 * Displays real-time execution logs
 */

import { useState, useEffect, useRef } from 'react';
import { useExecutionStream } from '../../hooks/use-execution-stream';

interface ExecutionLogsProps {
  workflowId: string;
}

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

export function ExecutionLogs({ workflowId }: ExecutionLogsProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState<'all' | 'info' | 'warning' | 'error' | 'success'>('all');
  const logsEndRef = useRef<HTMLDivElement>(null);

  useExecutionStream(workflowId, {
    onEvent: (event) => {
      const newLog: LogEntry = {
        id: `${Date.now()}-${Math.random()}`,
        timestamp: new Date(),
        level: 'info',
        message: `[${event.event}] ${JSON.stringify(event.data)}`,
      };
      setLogs((prev) => [...prev, newLog]);
    },
  });

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter((log) =>
    filter === 'all' || log.level === filter
  );

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'error':
        return 'text-red-600 dark:text-red-400';
      case 'warning':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'success':
        return 'text-green-600 dark:text-green-400';
      default:
        return 'text-gray-700 dark:text-gray-300';
    }
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700">
        <h3 className="text-sm font-medium text-gray-900 dark:text-white">
          Execution Logs
        </h3>

        <div className="flex items-center gap-2">
          {/* Filter */}
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="rounded border border-gray-300 px-2 py-1 text-xs dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          >
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
            <option value="success">Success</option>
          </select>

          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`text-xs ${
              autoScroll ? 'text-blue-600' : 'text-gray-500'
            } hover:text-blue-700 dark:text-blue-400`}
          >
            Auto-scroll: {autoScroll ? 'On' : 'Off'}
          </button>

          {/* Clear logs */}
          <button
            onClick={() => setLogs([])}
            className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Logs */}
      <div className="max-h-96 overflow-y-auto p-4">
        {filteredLogs.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
            No logs yet...
          </p>
        ) : (
          <div className="space-y-1 font-mono text-xs">
            {filteredLogs.map((log) => (
              <div key={log.id} className={getLevelColor(log.level)}>
                <span className="text-gray-500">
                  [{log.timestamp.toLocaleTimeString()}]
                </span>{' '}
                <span>{log.message}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
