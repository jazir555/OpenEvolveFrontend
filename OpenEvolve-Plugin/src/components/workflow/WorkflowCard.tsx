import { useState } from 'react';
import { useWorkflow } from '@/services/hooks/useWorkflows';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { cn } from '@/lib/utils';
import { BubbleButton } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface Workflow {
  evolution_id: string;
  name?: string;
  description?: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
  created_at?: string;
  started_at?: string;
  updated_at?: string;
  progress?: {
    current_iteration: number;
    max_iterations: number;
    percentage: number;
  };
}

interface WorkflowCardProps {
  workflow: Workflow;
  onClick?: (workflow: Workflow) => void;
  onExecute?: (evolutionId: string) => void;
  onPause?: (evolutionId: string) => void;
  onResume?: (evolutionId: string) => void;
  onStop?: (evolutionId: string) => void;
  onDelete?: (evolutionId: string) => void;
  className?: string;
}

function WorkflowCardBase({
  workflow,
  onClick,
  onExecute,
  onPause,
  onResume,
  onStop,
  onDelete,
  className,
}: WorkflowCardProps) {
  const [showActions, setShowActions] = useState(false);
  const formatDate = (value?: string) => {
    if (!value) return 'Unknown';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
  };
  const workflowId = workflow.evolution_id || (workflow as any).id || 'unknown';
  const createdDate = workflow.created_at || workflow.started_at;
  const updatedDate = workflow.updated_at || workflow.created_at || workflow.started_at;

  const handleAction = (e: React.MouseEvent, action: () => void) => {
    e.stopPropagation();
    action();
  };

  const canExecute = workflow.status === 'idle' || workflow.status === 'stopped' || workflow.status === 'failed';
  const canPause = workflow.status === 'running';
  const canResume = workflow.status === 'paused';
  const canStop = workflow.status === 'running' || workflow.status === 'paused';
  const canDelete = workflow.status === 'completed' || workflow.status === 'failed' || workflow.status === 'stopped';

  return (
    <div
      onClick={() => onClick?.(workflow)}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      className={cn(
        'workflow-card bg-white border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-all cursor-pointer',
        className
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            {workflow.name || `Workflow ${workflowId.slice(0, 8)}`}
          </h3>
          {workflow.description && (
            <p className="text-sm text-gray-600 line-clamp-2">{workflow.description}</p>
          )}
        </div>
        <StatusBadge status={workflow.status} />
      </div>

      <div className="space-y-2 mb-3">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Created</span>
          <span>{formatDate(createdDate)}</span>
        </div>
        <div className="flex justify-between text-xs text-gray-500">
          <span>Last Updated</span>
          <span>{formatDate(updatedDate)}</span>
        </div>
        {workflow.progress && (
          <div className="flex justify-between text-xs text-gray-500">
            <span>Progress</span>
            <span>
              {workflow.progress.current_iteration} / {workflow.progress.max_iterations}
            </span>
          </div>
        )}
      </div>

      {workflow.progress && (
        <div className="mb-3">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${workflow.progress.percentage}%` }}
            />
          </div>
        </div>
      )}

      <div
        className={cn(
          'flex gap-2 transition-opacity',
          showActions ? 'opacity-100' : 'opacity-0'
        )}
      >
        {canExecute && onExecute && (
          <BubbleButton
            onClick={(e) => handleAction(e, () => onExecute(workflowId))}
            className="px-3 py-1.5 text-xs"
          >
            Execute
          </BubbleButton>
        )}
        {canPause && onPause && (
          <BubbleButton
            onClick={(e) => handleAction(e, () => onPause(workflowId))}
            variant="secondary"
            className="px-3 py-1.5 text-xs"
          >
            Pause
          </BubbleButton>
        )}
        {canResume && onResume && (
          <BubbleButton
            onClick={(e) => handleAction(e, () => onResume(workflowId))}
            className="px-3 py-1.5 text-xs"
          >
            Resume
          </BubbleButton>
        )}
        {canStop && onStop && (
          <BubbleButton
            onClick={(e) => handleAction(e, () => onStop(workflowId))}
            variant="secondary"
            className="px-3 py-1.5 text-xs"
          >
            Stop
          </BubbleButton>
        )}
        {canDelete && onDelete && (
          <BubbleButton
            onClick={(e) => handleAction(e, () => onDelete(workflowId))}
            variant="ghost"
            className="px-3 py-1.5 text-xs"
          >
            Delete
          </BubbleButton>
        )}
      </div>
    </div>
  );
}

export const WorkflowCard = withComponentBoundary(WorkflowCardBase, 'WorkflowCard');
