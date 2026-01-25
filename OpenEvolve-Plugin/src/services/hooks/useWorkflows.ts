import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as api from '@/services/api/endpoints';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useCallback } from 'react';
import { errorLogger } from '@/utils';

/**
 * Workflow operations hook
 */
export function useWorkflows() {
  const queryClient = useQueryClient();
  const { setWorkflows, addWorkflow, updateWorkflow, removeWorkflow } = useWorkflowStore();

  // Fetch workflows list
  const { data, isLoading, error } = useQuery({
    queryKey: ['workflows'],
    queryFn: async () => {
      try {
        const response = await api.evolutionApi.list();
        const evolutions = response?.evolutions || [];
        setWorkflows(evolutions);
        return evolutions;
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          { component: 'useWorkflows', function: 'fetchWorkflows', additionalData: { queryKey: ['workflows'] } }
        );
        throw error;
      }
    },
    staleTime: 30000,
  });

  // Start workflow mutation
  const startWorkflow = useMutation({
    mutationFn: async (data: {
      content: string;
      mode: 'standard' | 'quality_diversity' | 'island_model';
      parameters: any;
      models: any[];
    }) => {
      try {
        const response = await api.evolutionApi.start(data);
        return response;
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          { component: 'useWorkflows', function: 'startWorkflow', additionalData: { data } }
        );
        throw error;
      }
    },
    onSuccess: (data) => {
      try {
        queryClient.invalidateQueries({ queryKey: ['workflows'] });
        if (data?.evolution_id) {
          const createdAt = data.created_at || new Date().toISOString();
          addWorkflow({
            evolution_id: data.evolution_id,
            status: (data.status as any) || 'idle',
            progress: {
              current_iteration: 0,
              max_iterations: 0,
              percentage: 0,
            },
            population: [],
            best_individual: null,
            metrics: {
              average_fitness: 0,
              diversity_score: 0,
              convergence_rate: 0,
            },
            started_at: createdAt,
            updated_at: createdAt,
            websocket_url: data.websocket_url,
          } as any);
        }
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          { component: 'useWorkflows', function: 'startWorkflow.onSuccess', additionalData: { data } }
        );
      }
    },
    onError: (error) => {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'useWorkflows', function: 'startWorkflow.onError' }
      );
    }
  });

  // Pause workflow mutation
  const pauseWorkflow = useMutation({
    mutationFn: async (evolutionId: string) => {
      const response = await api.evolutionApi.pause(evolutionId);
      updateWorkflow(evolutionId, { status: 'paused' });
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });

  // Resume workflow mutation
  const resumeWorkflow = useMutation({
    mutationFn: async (evolutionId: string) => {
      const response = await api.evolutionApi.resume(evolutionId);
      updateWorkflow(evolutionId, { status: 'running' });
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });

  // Stop workflow mutation
  const stopWorkflow = useMutation({
    mutationFn: async (evolutionId: string) => {
      const response = await api.evolutionApi.stop(evolutionId);
      updateWorkflow(evolutionId, { status: 'stopped' });
      
      // Auto-notify Knowledge Engine for extraction
      try {
        const workflow = await api.evolutionApi.getStatus(evolutionId);
        await api.knowledgeApi.notifyWorkflowComplete({
          workflow_id: evolutionId,
          problem_statement: workflow.input?.problem_statement || "Unknown problem",
          results: response.final_results || {}
        });
      } catch (e) {
        console.warn('Knowledge extraction trigger failed', e);
      }
      
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });

  // Delete workflow mutation
  const deleteWorkflowMutation = useMutation({
    mutationFn: async (evolutionId: string) => {
      await api.evolutionApi.delete(evolutionId);
      removeWorkflow(evolutionId);
      return evolutionId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });

  return {
    workflows: data || [],
    isLoading,
    error,
    startWorkflow: startWorkflow.mutateAsync,
    pauseWorkflow: pauseWorkflow.mutateAsync,
    resumeWorkflow: resumeWorkflow.mutateAsync,
    stopWorkflow: stopWorkflow.mutateAsync,
    deleteWorkflow: deleteWorkflowMutation.mutateAsync,
    isStarting: startWorkflow.isPending,
    isPausing: pauseWorkflow.isPending,
    isResuming: resumeWorkflow.isPending,
    isStopping: stopWorkflow.isPending,
    isDeleting: deleteWorkflowMutation.isPending,
  };
}

/**
 * Single workflow hook
 */
export function useWorkflow(evolutionId?: string) {
  const queryClient = useQueryClient();
  const { currentWorkflow, setCurrentWorkflow, updateWorkflow } = useWorkflowStore();

  // Fetch workflow details
  const { data, isLoading, error } = useQuery({
    queryKey: ['workflow', evolutionId],
    queryFn: async () => {
      if (!evolutionId) throw new Error('Evolution ID is required');
      const response = await api.evolutionApi.getStatus(evolutionId);
      setCurrentWorkflow(response);
      return response;
    },
    enabled: !!evolutionId,
    refetchInterval: 2000, // Poll every 2 seconds
  });

  // Pause workflow
  const pause = useCallback(async () => {
    if (!evolutionId) return;
    await api.evolutionApi.pause(evolutionId);
    updateWorkflow(evolutionId, { status: 'paused' });
    queryClient.invalidateQueries({ queryKey: ['workflow', evolutionId] });
  }, [evolutionId, updateWorkflow, queryClient]);

  // Resume workflow
  const resume = useCallback(async () => {
    if (!evolutionId) return;
    await api.evolutionApi.resume(evolutionId);
    updateWorkflow(evolutionId, { status: 'running' });
    queryClient.invalidateQueries({ queryKey: ['workflow', evolutionId] });
  }, [evolutionId, updateWorkflow, queryClient]);

  // Stop workflow
  const stop = useCallback(async () => {
    if (!evolutionId) return;
    await api.evolutionApi.stop(evolutionId);
    updateWorkflow(evolutionId, { status: 'stopped' });
    queryClient.invalidateQueries({ queryKey: ['workflow', evolutionId] });
  }, [evolutionId, updateWorkflow, queryClient]);

  // Delete workflow
  const remove = useCallback(async () => {
    if (!evolutionId) return;
    await api.evolutionApi.delete(evolutionId);
    setCurrentWorkflow(null);
    queryClient.invalidateQueries({ queryKey: ['workflows'] });
  }, [evolutionId, setCurrentWorkflow, queryClient]);

  return {
    workflow: data || currentWorkflow,
    isLoading,
    error,
    pause,
    resume,
    stop,
    remove,
    refetch: () => queryClient.invalidateQueries({ queryKey: ['workflow', evolutionId] }),
  };
}

/**
 * Workflow configuration hook
 */
export function useWorkflowConfig() {
  const { config, setConfig, resetConfig } = useWorkflowStore();

  const updateConfig = useCallback((updates: Partial<typeof config>) => {
    setConfig(updates);
  }, [setConfig]);

  const reset = useCallback(() => {
    resetConfig();
  }, [resetConfig]);

  return {
    config,
    updateConfig,
    reset,
  };
}

/**
 * Workflow models hook
 */
export function useWorkflowModels() {
  const { models, setModels, addModel, removeModel } = useWorkflowStore();

  const updateModels = useCallback((newModels: typeof models) => {
    setModels(newModels);
  }, [setModels]);

  const add = useCallback((model: any) => {
    addModel(model);
  }, [addModel]);

  const remove = useCallback((index: number) => {
    removeModel(index);
  }, [removeModel]);

  return {
    models,
    updateModels,
    add,
    remove,
  };
}

/**
 * Integrated workflow hook
 */
export function useIntegratedWorkflow() {
  const queryClient = useQueryClient();

  // Start integrated workflow
  const startWorkflow = useMutation({
    mutationFn: async (data: {
      problem_statement: string;
      workflow_template?: string;
      parameters?: Record<string, any>;
    }) => {
      const response = await api.workflowApi.start(data);
      return response;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });

  // Get workflow status
  const getWorkflowStatus = useCallback(async (workflowId: string) => {
    const response = await api.workflowApi.getStatus(workflowId);
    return response;
  }, []);

  return {
    startWorkflow: startWorkflow.mutateAsync,
    getWorkflowStatus,
    isStarting: startWorkflow.isPending,
  };
}
