import { useState } from 'react';
import { WorkflowCard, Workflow } from './WorkflowCard';
import { cn } from '@/lib/utils';
import { BubbleButton, BubbleInput } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

interface WorkflowListProps {
  workflows: Workflow[];
  onWorkflowSelect?: (workflow: Workflow) => void;
  onExecute?: (evolutionId: string) => void;
  onPause?: (evolutionId: string) => void;
  onResume?: (evolutionId: string) => void;
  onStop?: (evolutionId: string) => void;
  onDelete?: (evolutionId: string) => void;
  className?: string;
}

function WorkflowListBase({
  workflows = [],
  onWorkflowSelect,
  onExecute,
  onPause,
  onResume,
  onStop,
  onDelete,
  className,
}: WorkflowListProps) {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<'all' | 'running' | 'paused' | 'completed' | 'failed'>('all');

  const filteredWorkflows = workflows.filter((workflow) => {
    const workflowId = workflow.evolution_id || (workflow as any).id || '';
    const matchesSearch =
      (workflow.name?.toLowerCase() || '').includes(search.toLowerCase()) ||
      workflowId.toLowerCase().includes(search.toLowerCase());

    const matchesFilter = filter === 'all' || workflow.status === filter;

    return matchesSearch && matchesFilter;
  });

  return (
    <div className={cn('workflow-list', className)}>
      <div className="mb-4 space-y-3">
        <BubbleInput
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search workflows by name or ID..."
        />

        <div className="flex gap-2">
          {(['all', 'running', 'paused', 'completed', 'failed'] as const).map((status) => (
            <BubbleButton
              key={status}
              onClick={() => setFilter(status)}
              variant={filter === status ? 'primary' : 'secondary'}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </BubbleButton>
          ))}
        </div>
      </div>

      {filteredWorkflows.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <svg
            className="mx-auto h-12 w-12 text-gray-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          <p className="text-lg font-medium mb-1">No workflows found</p>
          <p className="text-sm">
            {search || filter !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Create your first workflow to get started'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredWorkflows.map((workflow) => (
            <WorkflowCard
              key={workflow.evolution_id || (workflow as any).id || workflow.name}
              workflow={workflow}
              onClick={onWorkflowSelect}
              onExecute={onExecute}
              onPause={onPause}
              onResume={onResume}
              onStop={onStop}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export const WorkflowList = withComponentBoundary(WorkflowListBase, 'WorkflowList');
