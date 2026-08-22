/**
 * Workflow Settings Hooks
 *
 * TanStack Query hooks for the Sovereign-Grade Decomposition Workflow settings
 * (`GET/PUT /workflows/{id}/settings`, plus create/run forwarding `settings`).
 *
 * Mirrors the query-key + ApiClient pattern used by `use-workflows-api.ts`,
 * but against the canonical `openevolveApi` client (the decomposition surface).
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { openevolveApi } from '@/services/openevolveApi';
import type {
  WorkflowCreate,
  WorkflowResponse,
  WorkflowSettings,
  WorkflowSettingsInput,
} from '@/services/openevolveApi';

// ============================================================================
// Query Keys
// ============================================================================

export const workflowSettingsKeys = {
  all: ['workflow-settings'] as const,
  detail: (id: string) => [...workflowSettingsKeys.all, id] as const,
};

// ============================================================================
// Read Settings
// ============================================================================

export function useWorkflowSettings(workflowId: string | undefined) {
  return useQuery({
    queryKey: workflowSettingsKeys.detail(workflowId ?? ''),
    queryFn: async () => {
      if (!workflowId) throw new Error('No workflow selected');
      return openevolveApi.getWorkflowSettings(workflowId);
    },
    enabled: !!workflowId,
  });
}

// ============================================================================
// Update Settings
// ============================================================================

export function useUpdateWorkflowSettings(workflowId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: WorkflowSettingsInput) => {
      if (!workflowId) throw new Error('No workflow selected');
      return openevolveApi.updateWorkflowSettings(workflowId, settings);
    },
    onSuccess: (data: WorkflowSettings) => {
      if (workflowId) {
        queryClient.setQueryData(workflowSettingsKeys.detail(workflowId), data);
      }
    },
  });
}

// ============================================================================
// Create Workflow (forwards settings)
// ============================================================================

export function useCreateWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      workflow,
      settings = {},
    }: {
      workflow: WorkflowCreate;
      settings?: WorkflowSettingsInput;
    }): Promise<WorkflowResponse> => {
      const created = await openevolveApi.createWorkflow(workflow, settings);
      return created;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });
}

// ============================================================================
// Run Workflow (forwards settings as config)
// ============================================================================

export function useRunWorkflow() {
  return useMutation({
    mutationFn: async ({
      workflowId,
      config = {},
    }: {
      workflowId: string;
      config?: WorkflowSettingsInput;
    }): Promise<Record<string, unknown>> => {
      return openevolveApi.runWorkflow(workflowId, config);
    },
  });
}
