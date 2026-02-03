/**
 * StatusBadge Component
 * Status indicator badge
 */

import { WorkflowStatus } from '../../types/api';

interface StatusBadgeProps {
  status: WorkflowStatus | string;
  text?: string;
}

export function StatusBadge({ status, text }: StatusBadgeProps) {
  const styles: Record<string, string> = {
    [WorkflowStatus.CREATED]: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    [WorkflowStatus.RUNNING]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    [WorkflowStatus.PAUSED]: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
    [WorkflowStatus.COMPLETED]: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    [WorkflowStatus.FAILED]: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    [WorkflowStatus.CANCELLED]: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  };

  const displayStatus = text || status;

  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${styles[status] || styles[WorkflowStatus.CREATED]}`}>
      {displayStatus.charAt(0).toUpperCase() + displayStatus.slice(1)}
    </span>
  );
}
