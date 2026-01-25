import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

type Status = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';

interface StatusBadgeProps {
  status?: Status;
  isConnected?: boolean;
  className?: string;
}

function StatusBadgeBase({ status, isConnected, className }: StatusBadgeProps) {
  const getStatusConfig = (statusValue?: Status) => {
    if (isConnected !== undefined) {
      return {
        color: isConnected ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800',
        dotColor: isConnected ? 'bg-green-500' : 'bg-gray-500',
        label: isConnected ? 'Connected' : 'Disconnected',
      };
    }

    const configs = {
      idle: {
        color: 'bg-gray-100 text-gray-800',
        dotColor: 'bg-gray-500',
        label: 'Idle',
      },
      running: {
        color: 'bg-blue-100 text-blue-800',
        dotColor: 'bg-blue-500',
        label: 'Running',
      },
      paused: {
        color: 'bg-yellow-100 text-yellow-800',
        dotColor: 'bg-yellow-500',
        label: 'Paused',
      },
      completed: {
        color: 'bg-green-100 text-green-800',
        dotColor: 'bg-green-500',
        label: 'Completed',
      },
      failed: {
        color: 'bg-red-100 text-red-800',
        dotColor: 'bg-red-500',
        label: 'Failed',
      },
      stopped: {
        color: 'bg-gray-100 text-gray-800',
        dotColor: 'bg-gray-500',
        label: 'Stopped',
      },
    };

    return configs[statusValue || 'idle'] || configs.idle;
  };

  const config = getStatusConfig(status);

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium',
        config.color,
        className
      )}
    >
      <span
        className={cn('h-2 w-2 rounded-full animate-pulse', config.dotColor)}
      />
      {config.label}
    </span>
  );
}

export const StatusBadge = withComponentBoundary(StatusBadgeBase, 'StatusBadge');
