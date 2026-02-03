/**
 * Workflow Store
 * Manages workflow state and operations
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  Workflow,
  WorkflowStatus,
  CreateWorkflowRequest,
  UpdateWorkflowRequest,
} from '../types/api';

// ============================================================================
// Workflow State Interface
// ============================================================================

interface WorkflowState {
  // Data
  workflows: Record<string, Workflow>;
  workflowIds: string[];
  selectedWorkflowId: string | null;

  // Loading states
  isLoading: boolean;
  isCreating: boolean;
  isUpdating: boolean;
  isDeleting: boolean;

  // Error state
  error: string | null;

  // Actions
  setWorkflows: (workflows: Workflow[]) => void;
  setWorkflow: (workflow: Workflow) => void;
  removeWorkflow: (workflowId: string) => void;
  setSelectedWorkflowId: (workflowId: string | null) => void;

  // Loading actions
  setIsLoading: (loading: boolean) => void;
  setIsCreating: (creating: boolean) => void;
  setIsUpdating: (updating: boolean) => void;
  setIsDeleting: (deleting: boolean) => void;

  // Error actions
  setError: (error: string | null) => void;
  clearError: () => void;

  // Reset
  reset: () => void;
}

// ============================================================================
// Create Store
// ============================================================================

export const useWorkflowStore = create<WorkflowState>()(
  subscribeWithSelector((set, get) => ({
    // Initial State
    workflows: {},
    workflowIds: [],
    selectedWorkflowId: null,
    isLoading: false,
    isCreating: false,
    isUpdating: false,
    isDeleting: false,
    error: null,

    // Set workflows (replace all)
    setWorkflows: (workflows) =>
      set(() => {
        const workflowMap: Record<string, Workflow> = {};
        const ids: string[] = [];

        for (const workflow of workflows) {
          workflowMap[workflow.id] = workflow;
          ids.push(workflow.id);
        }

        return {
          workflows: workflowMap,
          workflowIds: ids,
        };
      }),

    // Set single workflow (upsert)
    setWorkflow: (workflow) =>
      set((state) => {
        const workflows = { ...state.workflows };
        workflows[workflow.id] = workflow;

        const workflowIds = state.workflowIds.includes(workflow.id)
          ? state.workflowIds
          : [...state.workflowIds, workflow.id];

        return {
          workflows,
          workflowIds,
        };
      }),

    // Remove workflow
    removeWorkflow: (workflowId) =>
      set((state) => {
        const workflows = { ...state.workflows };
        delete workflows[workflowId];

        return {
          workflows,
          workflowIds: state.workflowIds.filter((id) => id !== workflowId),
          selectedWorkflowId:
            state.selectedWorkflowId === workflowId
              ? null
              : state.selectedWorkflowId,
        };
      }),

    // Set selected workflow
    setSelectedWorkflowId: (workflowId) =>
      set({ selectedWorkflowId: workflowId }),

    // Loading actions
    setIsLoading: (isLoading) => set({ isLoading }),
    setIsCreating: (isCreating) => set({ isCreating }),
    setIsUpdating: (isUpdating) => set({ isUpdating }),
    setIsDeleting: (isDeleting) => set({ isDeleting }),

    // Error actions
    setError: (error) => set({ error }),
    clearError: () => set({ error: null }),

    // Reset
    reset: () =>
      set({
        workflows: {},
        workflowIds: [],
        selectedWorkflowId: null,
        isLoading: false,
        isCreating: false,
        isUpdating: false,
        isDeleting: false,
        error: null,
      }),
  }))
);

// ============================================================================
// Selectors
// ============================================================================

/**
 * Get workflow by ID
 */
export const getWorkflowById = (id: string) => {
  return useWorkflowStore.getState().workflows[id];
};

/**
 * Get all workflows
 */
export const getAllWorkflows = () => {
  const state = useWorkflowStore.getState();
  return state.workflowIds.map((id) => state.workflows[id]);
};

/**
 * Get workflows by status
 */
export const getWorkflowsByStatus = (status: WorkflowStatus) => {
  const workflows = getAllWorkflows();
  return workflows.filter((w) => w.status === status);
};

/**
 * Get selected workflow
 */
export const getSelectedWorkflow = () => {
  const state = useWorkflowStore.getState();
  if (!state.selectedWorkflowId) return null;
  return state.workflows[state.selectedWorkflowId];
};

/**
 * Get running workflows
 */
export const getRunningWorkflows = () => {
  return getWorkflowsByStatus(WorkflowStatus.RUNNING);
};

/**
 * Get completed workflows
 */
export const getCompletedWorkflows = () => {
  return getWorkflowsByStatus(WorkflowStatus.COMPLETED);
};

/**
 * Get failed workflows
 */
export const getFailedWorkflows = () => {
  return getWorkflowsByStatus(WorkflowStatus.FAILED);
};

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook to get all workflows
 */
export const useWorkflows = () => {
  return useWorkflowStore((state) => {
    return {
      workflows: state.workflowIds.map((id) => state.workflows[id]),
      isLoading: state.isLoading,
      error: state.error,
    };
  });
};

/**
 * Hook to get a specific workflow
 */
export const useWorkflow = (workflowId: string) => {
  return useWorkflowStore((state) => {
    return {
      workflow: state.workflows[workflowId] || null,
      isLoading: state.isLoading,
      error: state.error,
    };
  });
};

/**
 * Hook to get selected workflow
 */
export const useSelectedWorkflow = () => {
  const selectedWorkflowId = useWorkflowStore((state) => state.selectedWorkflowId);
  const workflow = useWorkflowStore((state) =>
    state.selectedWorkflowId ? state.workflows[state.selectedWorkflowId] : null
  );

  return {
    workflow,
    selectedWorkflowId,
    setSelectedWorkflowId: useWorkflowStore((state) => state.setSelectedWorkflowId),
  };
};

/**
 * Hook to get running workflows
 */
export const useRunningWorkflows = () => {
  const workflows = useWorkflowStore((state) => {
    return state.workflowIds.map((id) => state.workflows[id]);
  });

  return workflows.filter((w) => w.status === WorkflowStatus.RUNNING);
};

/**
 * Hook to get workflow statistics
 */
export const useWorkflowStats = () => {
  const workflows = useWorkflowStore((state) => {
    return state.workflowIds.map((id) => state.workflows[id]);
  });

  return {
    total: workflows.length,
    running: workflows.filter((w) => w.status === WorkflowStatus.RUNNING).length,
    completed: workflows.filter((w) => w.status === WorkflowStatus.COMPLETED).length,
    failed: workflows.filter((w) => w.status === WorkflowStatus.FAILED).length,
    created: workflows.filter((w) => w.status === WorkflowStatus.CREATED).length,
  };
};
