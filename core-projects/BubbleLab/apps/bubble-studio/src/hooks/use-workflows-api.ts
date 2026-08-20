/**
 * Workflow API Hooks
 * React hooks for workflow operations
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import apiClient, { setAuthToken, clearAuthToken } from '../lib/api-client';
import { useWorkflowStore } from '../stores/workflowStore';
import {
  Workflow,
  CreateWorkflowRequest,
  UpdateWorkflowRequest,
  ListQueryParams,
} from '../types/api';

// ============================================================================
// Query Keys
// ============================================================================

export const workflowKeys = {
  all: ['workflows'] as const,
  lists: () => [...workflowKeys.all, 'list'] as const,
  list: (params: ListQueryParams) => [...workflowKeys.lists(), params] as const,
  details: () => [...workflowKeys.all, 'detail'] as const,
  detail: (id: string) => [...workflowKeys.details(), id] as const,
};

// ============================================================================
// Fetch Workflows
// ============================================================================

export function useWorkflows(params?: ListQueryParams) {
  const setWorkflows = useWorkflowStore((state) => state.setWorkflows);
  const setIsLoading = useWorkflowStore((state) => state.setIsLoading);
  const setError = useWorkflowStore((state) => state.setError);

  return useQuery({
    queryKey: workflowKeys.list(params || {}),
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await apiClient.getWorkflows(params);
        setWorkflows(response.workflows);
        return response.workflows;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch workflows';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
  });
}

export function useWorkflow(workflowId: string) {
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);
  const setIsLoading = useWorkflowStore((state) => state.setIsLoading);
  const setError = useWorkflowStore((state) => state.setError);

  return useQuery({
    queryKey: workflowKeys.detail(workflowId),
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const workflow = await apiClient.getWorkflow(workflowId);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch workflow';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    enabled: !!workflowId,
  });
}

// ============================================================================
// Create Workflow
// ============================================================================

export function useCreateWorkflow() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const setIsCreating = useWorkflowStore((state) => state.setIsCreating);
  const setError = useWorkflowStore((state) => state.setError);
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);

  return useMutation({
    mutationFn: async (data: CreateWorkflowRequest) => {
      setIsCreating(true);
      setError(null);
      try {
        const workflow = await apiClient.createWorkflow(data);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to create workflow';
        setError(message);
        throw error;
      } finally {
        setIsCreating(false);
      }
    },
    onSuccess: (workflow: any) => {
      // Invalidate workflows list query
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });

      // Navigate to workflow detail page
      navigate({
        to: '/oe-workflows/$workflowId',
        params: { workflowId: workflow.id },
      });
    },
  });
}

// ============================================================================
// Update Workflow
// ============================================================================

/**
 * NOT SUPPORTED BY THE BACKEND: there is no `PUT /api/workflows/{id}` route, so
 * `apiClient.updateWorkflow` rejects immediately with an `UnsupportedEndpointError`
 * (surfaced through `setError`) instead of issuing a request. Kept for API
 * stability; use `openevolveApi.updateWorkflowPlan` to edit a decomposition plan.
 */
export function useUpdateWorkflow() {
  const queryClient = useQueryClient();
  const setIsUpdating = useWorkflowStore((state) => state.setIsUpdating);
  const setError = useWorkflowStore((state) => state.setError);
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);

  return useMutation({
    mutationFn: async ({ workflowId, data }: { workflowId: string; data: UpdateWorkflowRequest }) => {
      setIsUpdating(true);
      setError(null);
      try {
        const workflow = await apiClient.updateWorkflow(workflowId, data);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to update workflow';
        setError(message);
        throw error;
      } finally {
        setIsUpdating(false);
      }
    },
    onSuccess: (_: any, variables: any) => {
      // Invalidate specific workflow query
      queryClient.invalidateQueries({
        queryKey: workflowKeys.detail(variables.workflowId),
      });

      // Invalidate workflows list query
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
    },
  });
}

// ============================================================================
// Delete Workflow
// ============================================================================

export function useDeleteWorkflow() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const setIsDeleting = useWorkflowStore((state) => state.setIsDeleting);
  const setError = useWorkflowStore((state) => state.setError);
  const removeWorkflow = useWorkflowStore((state) => state.removeWorkflow);

  return useMutation({
    mutationFn: async (workflowId: string) => {
      setIsDeleting(true);
      setError(null);
      try {
        await apiClient.deleteWorkflow(workflowId);
        removeWorkflow(workflowId);
        return workflowId;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to delete workflow';
        setError(message);
        throw error;
      } finally {
        setIsDeleting(false);
      }
    },
    onSuccess: () => {
      // Invalidate workflows list query
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });

      // Navigate back to workflows list
      navigate({ to: '/oe-workflows' });
    },
  });
}

// ============================================================================
// Workflow Execution Actions
// ============================================================================

/**
 * NOT SUPPORTED BY THE BACKEND: there is no `POST /api/workflows/{id}/start` route.
 * `apiClient.startWorkflow` rejects immediately (message surfaced via `setError`).
 * Launch a run with `openevolveApi.executeWorkflow` (`POST /api/executions`) instead.
 * Kept exported so existing imports keep working; the UI
 * (`components/execution/ExecutionControls.tsx`) renders Start disabled.
 */
export function useStartWorkflow() {
  const queryClient = useQueryClient();
  const setError = useWorkflowStore((state) => state.setError);
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);

  return useMutation({
    mutationFn: async (workflowId: string) => {
      setError(null);
      try {
        const workflow = await apiClient.startWorkflow(workflowId);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to start workflow';
        setError(message);
        throw error;
      }
    },
    onSuccess: (_: any, workflowId: any) => {
      // Invalidate workflow query
      queryClient.invalidateQueries({
        queryKey: workflowKeys.detail(workflowId),
      });

      // Invalidate workflows list
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
    },
  });
}

export function usePauseWorkflow() {
  const queryClient = useQueryClient();
  const setError = useWorkflowStore((state) => state.setError);
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);

  return useMutation({
    mutationFn: async (workflowId: string) => {
      setError(null);
      try {
        const workflow = await apiClient.pauseWorkflow(workflowId);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to pause workflow';
        setError(message);
        throw error;
      }
    },
    onSuccess: (_: any, workflowId: any) => {
      queryClient.invalidateQueries({
        queryKey: workflowKeys.detail(workflowId),
      });
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
    },
  });
}

export function useResumeWorkflow() {
  const queryClient = useQueryClient();
  const setError = useWorkflowStore((state) => state.setError);
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);

  return useMutation({
    mutationFn: async (workflowId: string) => {
      setError(null);
      try {
        const workflow = await apiClient.resumeWorkflow(workflowId);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to resume workflow';
        setError(message);
        throw error;
      }
    },
    onSuccess: (_: any, workflowId: any) => {
      queryClient.invalidateQueries({
        queryKey: workflowKeys.detail(workflowId),
      });
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
    },
  });
}

/**
 * NOT SUPPORTED BY THE BACKEND: there is no `POST /api/workflows/{id}/stop` route.
 * `apiClient.stopWorkflow` rejects immediately (message surfaced via `setError`).
 * Cancel the underlying run with `openevolveApi.cancelExecution`
 * (`POST /api/executions/{id}/cancel`) instead. Kept exported for API stability;
 * the UI renders Stop disabled.
 */
export function useStopWorkflow() {
  const queryClient = useQueryClient();
  const setError = useWorkflowStore((state) => state.setError);
  const setWorkflow = useWorkflowStore((state) => state.setWorkflow);

  return useMutation({
    mutationFn: async (workflowId: string) => {
      setError(null);
      try {
        const workflow = await apiClient.stopWorkflow(workflowId);
        setWorkflow(workflow);
        return workflow;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to stop workflow';
        setError(message);
        throw error;
      }
    },
    onSuccess: (_: any, workflowId: any) => {
      queryClient.invalidateQueries({
        queryKey: workflowKeys.detail(workflowId),
      });
      queryClient.invalidateQueries({ queryKey: workflowKeys.lists() });
    },
  });
}

export function useWorkflowResults(workflowId: string) {
  const setIsLoading = useWorkflowStore((state) => state.setIsLoading);
  const setError = useWorkflowStore((state) => state.setError);

  return useQuery({
    queryKey: ['workflow-results', workflowId],
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const results = await apiClient.getWorkflowResults(workflowId);
        return results;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch results';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    enabled: !!workflowId,
  });
}
