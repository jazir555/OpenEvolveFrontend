/**
 * Create Workflow Page
 * New workflow creation wizard
 */

import { createFileRoute } from '@tanstack/react-router';
import { WorkflowConfigForm } from '../components/workflow/WorkflowConfigForm';

export const Route = createFileRoute('/oe-workflows/create')({
  component: CreateWorkflowPage,
});

function CreateWorkflowPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Create Workflow
        </h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Configure a new OpenEvolve workflow step by step
        </p>
      </div>

      {/* Workflow Configuration Form */}
      <WorkflowConfigForm />
    </div>
  );
}
