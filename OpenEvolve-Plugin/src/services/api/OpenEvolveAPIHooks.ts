/**
 * React Hooks for OpenEvolve API
 *
 * This file provides custom React hooks that wrap the OpenEvolve API service
 * for easy integration into React components.
 *
 * @module OpenEvolveAPIHooks
 */

import { useState, useEffect, useCallback } from 'react';
import {
  openEvolveAPI,
  EvolutionRun,
  EvolutionConfig,
  AdversarialRun,
  AdversarialConfig,
  KnowledgeEntry,
  KnowledgeCategory,
  KnowledgeStats,
  KnowledgeQueryParams,
  WorkflowDefinition,
  WorkflowInstance,
  WorkflowPerformance,
  TeamPerformance,
  GauntletPerformance,
  SolutionQuality,
  AnalyticsQueryParams,
  DecompositionProblem,
  SubProblem,
} from './OpenEvolveAPI';

// ============================================================================
// GENERIC HOOK TYPES
// ============================================================================

interface UseApiResult<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

interface UseMutationResult<TData, TParams> {
  data: TData | null;
  isLoading: boolean;
  error: Error | null;
  mutate: (params: TParams) => Promise<void>;
  reset: () => void;
}

// ============================================================================
// EVOLUTION HOOKS
// ============================================================================

/**
 * Hook for fetching all evolution runs
 */
export function useEvolutionRuns(params?: { status?: string; limit?: number }) {
  const [data, setData] = useState<EvolutionRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getEvolutionRuns(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching evolution runs:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching a single evolution run
 */
export function useEvolutionRun(runId: string) {
  const [data, setData] = useState<EvolutionRun | null>(null);
  const [isLoading, setIsLoading] = useState(!!runId);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!runId) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getEvolutionRun(runId);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching evolution run:', err);
    } finally {
      setIsLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for creating evolution runs
 */
export function useCreateEvolutionRun() {
  const [data, setData] = useState<EvolutionRun | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = useCallback(async (params: { name: string; config: EvolutionConfig }) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.createEvolutionRun(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, isLoading, error, mutate, reset };
}

/**
 * Hook for evolution configuration
 */
export function useEvolutionConfig() {
  const [data, setData] = useState<EvolutionConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getEvolutionConfig();
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching evolution config:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const updateConfig = useCallback(async (config: Partial<EvolutionConfig>) => {
    try {
      const result = await openEvolveAPI.updateEvolutionConfig(config);
      setData(result);
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, []);

  return { data, isLoading, error, refetch: fetchData, updateConfig };
}

// ============================================================================
// ADVERSARIAL HOOKS
// ============================================================================

/**
 * Hook for fetching all adversarial runs
 */
export function useAdversarialRuns(params?: { status?: string; limit?: number }) {
  const [data, setData] = useState<AdversarialRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getAdversarialRuns(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching adversarial runs:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching a single adversarial run
 */
export function useAdversarialRun(runId: string) {
  const [data, setData] = useState<AdversarialRun | null>(null);
  const [isLoading, setIsLoading] = useState(!!runId);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!runId) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getAdversarialRun(runId);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching adversarial run:', err);
    } finally {
      setIsLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for creating adversarial runs
 */
export function useCreateAdversarialRun() {
  const [data, setData] = useState<AdversarialRun | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = useCallback(async (params: { name: string; config: AdversarialConfig }) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.createAdversarialRun(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, isLoading, error, mutate, reset };
}

/**
 * Hook for adversarial configuration
 */
export function useAdversarialConfig() {
  const [data, setData] = useState<AdversarialConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getAdversarialConfig();
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching adversarial config:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const updateConfig = useCallback(async (config: Partial<AdversarialConfig>) => {
    try {
      const result = await openEvolveAPI.updateAdversarialConfig(config);
      setData(result);
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, []);

  return { data, isLoading, error, refetch: fetchData, updateConfig };
}

// ============================================================================
// KNOWLEDGE BASE HOOKS
// ============================================================================

/**
 * Hook for fetching knowledge entries
 */
export function useKnowledgeEntries(params?: KnowledgeQueryParams) {
  const [data, setData] = useState<KnowledgeEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getKnowledgeEntries(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching knowledge entries:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching knowledge categories
 */
export function useKnowledgeCategories() {
  const [data, setData] = useState<KnowledgeCategory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getKnowledgeCategories();
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching knowledge categories:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching knowledge statistics
 */
export function useKnowledgeStats() {
  const [data, setData] = useState<KnowledgeStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getKnowledgeStats();
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching knowledge stats:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for creating knowledge entries
 */
export function useCreateKnowledgeEntry() {
  const [data, setData] = useState<KnowledgeEntry | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = useCallback(async (params: {
    title: string;
    content: string;
    category: string;
    tags: string[];
    status?: 'draft' | 'published' | 'archived';
  }) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.createKnowledgeEntry(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, isLoading, error, mutate, reset };
}

// ============================================================================
// WORKFLOW HOOKS
// ============================================================================

/**
 * Hook for fetching workflows
 */
export function useWorkflows(params?: { status?: string; limit?: number }) {
  const [data, setData] = useState<WorkflowDefinition[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getWorkflows(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching workflows:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching a single workflow
 */
export function useWorkflow(workflowId: string) {
  const [data, setData] = useState<WorkflowDefinition | null>(null);
  const [isLoading, setIsLoading] = useState(!!workflowId);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!workflowId) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getWorkflow(workflowId);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching workflow:', err);
    } finally {
      setIsLoading(false);
    }
  }, [workflowId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for workflow instances
 */
export function useWorkflowInstances(workflowId: string) {
  const [data, setData] = useState<WorkflowInstance[]>([]);
  const [isLoading, setIsLoading] = useState(!!workflowId);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!workflowId) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getWorkflowInstances(workflowId);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching workflow instances:', err);
    } finally {
      setIsLoading(false);
    }
  }, [workflowId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for workflow templates
 */
export function useWorkflowTemplates() {
  const [data, setData] = useState<WorkflowDefinition[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getWorkflowTemplates();
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching workflow templates:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

// ============================================================================
// ANALYTICS HOOKS
// ============================================================================

/**
 * Hook for fetching workflow performance data
 */
export function useWorkflowPerformance(params?: AnalyticsQueryParams) {
  const [data, setData] = useState<WorkflowPerformance[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getWorkflowPerformance(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching workflow performance:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching team performance data
 */
export function useTeamPerformance(params?: AnalyticsQueryParams) {
  const [data, setData] = useState<TeamPerformance[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getTeamPerformance(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching team performance:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching gauntlet performance data
 */
export function useGauntletPerformance(params?: AnalyticsQueryParams) {
  const [data, setData] = useState<GauntletPerformance[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getGauntletPerformance(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching gauntlet performance:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching solution quality data
 */
export function useSolutionQuality(params?: AnalyticsQueryParams) {
  const [data, setData] = useState<SolutionQuality[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getSolutionQuality(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching solution quality:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching comprehensive analytics overview
 */
export function useAnalyticsOverview(params?: AnalyticsQueryParams) {
  const [data, setData] = useState<{
    workflows: WorkflowPerformance[];
    teams: TeamPerformance[];
    gauntlets: GauntletPerformance[];
    solutions: SolutionQuality[];
    knowledge: KnowledgeStats;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getAnalyticsOverview(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching analytics overview:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

// ============================================================================
// DECOMPOSITION HOOKS
// ============================================================================

/**
 * Hook for fetching decomposition problems
 */
export function useDecompositionProblems(params?: { status?: string; limit?: number }) {
  const [data, setData] = useState<DecompositionProblem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getDecompositionProblems(params);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching decomposition problems:', err);
    } finally {
      setIsLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching sub-problems
 */
export function useSubProblems(problemId: string) {
  const [data, setData] = useState<SubProblem[]>([]);
  const [isLoading, setIsLoading] = useState(!!problemId);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!problemId) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await openEvolveAPI.getSubProblems(problemId);
      setData(result);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching sub-problems:', err);
    } finally {
      setIsLoading(false);
    }
  }, [problemId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
}

// ============================================================================
// SYSTEM HOOKS
// ============================================================================

/**
 * Hook for fetching system health status
 */
export function useHealthStatus(pollInterval?: number) {
  const [data, setData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setError(null);
    try {
      const result = await openEvolveAPI.getHealthStatus();
      setData(result);
      setIsLoading(false);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching health status:', err);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();

    if (pollInterval) {
      const interval = setInterval(fetchData, pollInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, pollInterval]);

  return { data, isLoading, error, refetch: fetchData };
}

/**
 * Hook for fetching system status
 */
export function useSystemStatus(pollInterval?: number) {
  const [data, setData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setError(null);
    try {
      const result = await openEvolveAPI.getSystemStatus();
      setData(result);
      setIsLoading(false);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching system status:', err);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();

    if (pollInterval) {
      const interval = setInterval(fetchData, pollInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, pollInterval]);

  return { data, isLoading, error, refetch: fetchData };
}
