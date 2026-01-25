import { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface LogEntry {
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  source?: string;
}

interface LiveLogViewerProps {
  logs: LogEntry[];
  className?: string;
  maxHeight?: string;
}

function LiveLogViewerBase({ logs, className, maxHeight = '24rem' }: LiveLogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'error':
        return 'text-red-400';
      case 'warn':
        return 'text-yellow-400';
      case 'info':
        return 'text-blue-400';
      case 'debug':
        return 'text-gray-400';
      default:
        return 'text-gray-300';
    }
  };

  const getLevelIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'error':
        return '✖';
      case 'warn':
        return '⚠';
      case 'info':
        return 'ℹ';
      case 'debug':
        return '▸';
      default:
        return '•';
    }
  };

  return (
    <div
      className={cn(
        'log-viewer overflow-y-auto font-mono text-sm bg-gray-900 text-gray-100 p-4 rounded-lg',
        className
      )}
      style={{ maxHeight }}
    >
      {logs.length === 0 ? (
        <div className="text-gray-500 text-center py-8">
          No logs available. Waiting for events...
        </div>
      ) : (
        <div className="space-y-1">
          {logs.map((log, i) => (
            <div key={i} className="flex items-start gap-2 hover:bg-gray-800 px-2 py-1 rounded">
              <span className="text-gray-600 text-xs whitespace-nowrap">
                {log.timestamp}
              </span>
              <span className={cn('font-semibold text-xs', getLevelColor(log.level))}>
                [{getLevelIcon(log.level)} {log.level.toUpperCase()}]
              </span>
              {log.source && (
                <span className="text-purple-400 text-xs">[{log.source}]</span>
              )}
              <span className="flex-1 break-words">{log.message}</span>
            </div>
          ))}
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

export const LiveLogViewer = withComponentBoundary(LiveLogViewerBase, 'LiveLogViewer');
