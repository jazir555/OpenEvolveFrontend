// @ts-nocheck
import { useState, useCallback, useEffect, useRef } from 'react';
import { apiClient } from '../services/api/client';
import { useWorkflowStore } from '../stores/workflowStore';

/**
 * Decomposition parameters
 */
export interface DecompositionParams {
  problem_statement: string;
  decomposition_strategy?: 'hierarchical' | 'functional' | 'data_flow' | 'hybrid';
  max_depth?: number;
  include_dependencies?: boolean;
  include_subtasks?: boolean;
}

/**
 * Decomposition result
 */
export interface DecompositionResult {
  decomposition_id: string;
  problem_statement: string;
  decomposition_tree: DecompositionNode;
  tasks: DecompositionTask[];
  dependencies: Dependency[];
  execution_plan: ExecutionStep[];
  metadata: {
    total_tasks: number;
    estimated_duration: number;
    complexity_score: number;
    created_at: string;
  };
}

/**
 * Decomposition node in tree
 */
export interface DecompositionNode {
  id: string;
  title: string;
  description: string;
  level: number;
  children: DecompositionNode[];
  metadata?: Record<string, any>;
}

/**
 * Individual task
 */
export interface DecompositionTask {
  task_id: string;
  title: string;
  description: string;
  parent_id?: string;
  dependencies: string[];
  status: 'pending' | 'in_progress' | 'completed' | 'blocked';
  estimated_effort: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignee?: string;
}

/**
 * Task dependency
 */
export interface Dependency {
  from: string;
  to: string;
  type: 'sequential' | 'parallel' | 'conditional';
}

/**
 * Execution step
 */
export interface ExecutionStep {
  step_id: string;
  task_id: string;
  order: number;
  can_start_in_parallel: boolean;
}

/**
 * Decomposition state
 */
export interface DecompositionState {
  data: DecompositionResult | null;
  loading: boolean;
  error: Error | null;
  progress: number;
}

/**
 * Custom hook for problem decomposition
 * Breaks down complex problems into manageable subtasks
 */
export function useDecomposition(decompositionId?: string) {
  const [state, setState] = useState<DecompositionState>({
    data: null,
    loading: false,
    error: null,
    progress: 0,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const { setLoading, setError } = useWorkflowStore();

  /**
   * Execute decomposition
   */
  const execute = useCallback(async (params: DecompositionParams): Promise<DecompositionResult | null> => {
    setState(prev => ({ ...prev, loading: true, error: null, progress: 0 }));
    setLoading(true);
    setError(null);

    try {
      abortControllerRef.current = new AbortController();

      const response = await apiClient.post<DecompositionResult>(
        '/decomposition/analyze',
        params
      );

      setState(prev => ({
        ...prev,
        data: response,
        loading: false,
        progress: 100,
      }));

      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Get decomposition status
   */
  const getStatus = useCallback(async (): Promise<DecompositionResult | null> => {
    if (!decompositionId) {
      return null;
    }

    try {
      const status = await apiClient.get<DecompositionResult>(
        `/decomposition/${decompositionId}`
      );

      setState(prev => ({ ...prev, data: status }));
      return status;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error }));
      setError(error.message);
      return null;
    }
  }, [decompositionId, setError]);

  /**
   * Get decomposition results
   */
  const getResults = useCallback((): DecompositionResult | null => {
    return state.data;
  }, [state.data]);

  /**
   * Cancel decomposition
   */
  const cancel = useCallback(async (): Promise<void> => {
    setState(prev => ({ ...prev, loading: false, progress: 0 }));
    abortControllerRef.current?.abort();
    setLoading(false);
  }, [setLoading]);

  /**
   * Update task status
   */
  const updateTaskStatus = useCallback(async (
    taskId: string,
    status: DecompositionTask['status']
  ): Promise<void> => {
    if (!state.data) {
      return;
    }

    try {
      await apiClient.patch(`/decomposition/${state.data.decomposition_id}/tasks/${taskId}`, {
        status,
      });

      setState(prev => {
        if (!prev.data) return prev;

        const updatedTasks = prev.data.tasks.map(task =>
          task.task_id === taskId ? { ...task, status } : task
        );

        return {
          ...prev,
          data: {
            ...prev.data,
            tasks: updatedTasks,
          },
        };
      });
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
    }
  }, [state.data, setError]);

  /**
   * Get execution plan
   */
  const getExecutionPlan = useCallback((): ExecutionStep[] => {
    return state.data?.execution_plan || [];
  }, [state.data]);

  /**
   * Get task by ID
   */
  const getTaskById = useCallback((taskId: string): DecompositionTask | null => {
    return state.data?.tasks.find(task => task.task_id === taskId) || null;
  }, [state.data]);

  /**
   * Get tasks by status
   */
  const getTasksByStatus = useCallback((status: DecompositionTask['status']): DecompositionTask[] => {
    return state.data?.tasks.filter(task => task.status === status) || [];
  }, [state.data]);

  /**
   * Reset state
   */
  const reset = useCallback((): void => {
    setState({
      data: null,
      loading: false,
      error: null,
      progress: 0,
    });
    setError(null);
  }, [setError]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return {
    ...state,
    execute,
    getStatus,
    getResults,
    cancel,
    updateTaskStatus,
    getExecutionPlan,
    getTaskById,
    getTasksByStatus,
    reset,
  };
}

/**
 * Decomposition history hook
 */
export function useDecompositionHistory(params?: {
  limit?: number;
  offset?: number;
}) {
  const [state, setState] = useState<{
    data: DecompositionResult[] | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const fetchHistory = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await apiClient.get<{ decompositions: DecompositionResult[] }>(
        '/decomposition',
        params
      );

      setState(prev => ({
        ...prev,
        data: response?.decompositions || [],
        loading: false,
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, [params]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return {
    ...state,
    refetch: fetchHistory,
  };
}

/**
 * Decomposition templates hook
 */
export function useDecompositionTemplates() {
  const [state, setState] = useState<{
    data: Array<{ id: string; name: string; description: string; strategy: string }> | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const fetchTemplates = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await apiClient.get<{ templates: any[] }>(
        '/decomposition/templates'
      );

      setState(prev => ({
        ...prev,
        data: response?.templates || [],
        loading: false,
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, []);

  useEffect(() => {
    fetchTemplates();
  }, [fetchTemplates]);

  return {
    ...state,
    refetch: fetchTemplates,
  };
}
