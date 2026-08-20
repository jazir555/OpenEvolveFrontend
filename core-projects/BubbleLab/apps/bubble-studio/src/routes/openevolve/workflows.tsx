import { createFileRoute } from '@tanstack/react-router';
import { WorkflowOrchestrator } from '@/components/workflows';

export const Route = createFileRoute('/openevolve/workflows')({
  component: OpenEvolveWorkflowsPage,
});

function OpenEvolveWorkflowsPage() {
  return (
    <div className="p-6">
      <WorkflowOrchestrator />
    </div>
  );
}
