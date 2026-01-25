// @ts-nocheck
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/services/api/client';
import * as api from '@/services/api/endpoints';
import { useAuthStore } from '@/stores/authStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useCallback, useEffect } from 'react';
// @ts-nocheck

/**
 * Generic API call hook
 */
export function useApi() {
  const queryClient = useQueryClient();
  const token = useAuthStore((state) => state.token);

  /**
   * Generic GET request hook
   */
  const useGet = <T>(
    queryKey: string[],
    fn: () => Promise<T>,
    options?: {
      enabled?: boolean;
      refetchInterval?: number;
      staleTime?: number;
    }
  ) => {
    return useQuery({
      queryKey,
      queryFn: fn,
      enabled: !!token && (options?.enabled !== false),
      refetchInterval: options?.refetchInterval,
      staleTime: options?.staleTime,
    });
  };

  /**
   * Generic POST mutation hook
   */
  const usePost = <T, V>(
    fn: (data: V) => Promise<T>,
    options?: {
      onSuccess?: (data: T, variables: V) => void;
      onError?: (error: any) => void;
    }
  ) => {
    return useMutation({
      mutationFn: fn,
      onSuccess: options?.onSuccess,
      onError: options?.onError,
    });
  };

  return {
    useGet,
    usePost,
    queryClient,
    isAuthenticated: !!token,
  };
}

/**
 * Auth hooks
 */
export function useAuth() {
  const {
    user,
    token,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    register,
    updateUser,
    clearError,
  } = useAuthStore();

  return {
    user,
    token,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    register,
    updateUser,
    clearError,
  };
}

/**
 * Evolution hooks
 */
export function useEvolution(evolutionId?: string) {
  const { useGet } = useApi();

  // Get evolution status
  const statusQuery = useGet(
    ['evolution', evolutionId],
    () => api.evolutionApi.getStatus(evolutionId!),
    {
      enabled: !!evolutionId,
      refetchInterval: 2000, // Poll every 2 seconds
    }
  );

  // Start evolution mutation
  const startMutation = useMutation({
    mutationFn: ({
      content,
      mode,
      parameters,
      models,
    }: {
      content: string;
      mode: 'standard' | 'quality_diversity' | 'island_model';
      parameters: any;
      models: any[];
    }) => api.evolutionApi.start({ content, mode, parameters, models }),
  });

  // Pause evolution mutation
  const pauseMutation = useMutation({
    mutationFn: (id: string) => api.evolutionApi.pause(id),
  });

  // Resume evolution mutation
  const resumeMutation = useMutation({
    mutationFn: (id: string) => api.evolutionApi.resume(id),
  });

  // Stop evolution mutation
  const stopMutation = useMutation({
    mutationFn: (id: string) => api.evolutionApi.stop(id),
  });

  return {
    evolution: statusQuery.data,
    isLoading: statusQuery.isLoading,
    error: statusQuery.error,
    refetch: statusQuery.refetch,
    startEvolution: startMutation.mutateAsync,
    pauseEvolution: pauseMutation.mutateAsync,
    resumeEvolution: resumeMutation.mutateAsync,
    stopEvolution: stopMutation.mutateAsync,
    isStarting: startMutation.isPending,
    isPausing: pauseMutation.isPending,
    isResuming: resumeMutation.isPending,
    isStopping: stopMutation.isPending,
  };
}

/**
 * Evolution list hook
 */
export function useEvolutions(params?: {
  status?: string;
  limit?: number;
  offset?: number;
}) {
  const { useGet } = useApi();

  return useGet(
    ['evolutions', params],
    () => api.evolutionApi.list(params),
    {
      staleTime: 30000, // 30 seconds
    }
  );
}

/**
 * Adversarial testing hooks
 */
export function useAdversarialTest(testId?: string) {
  const { useGet } = useApi();

  // Get test status
  const statusQuery = useGet(
    ['adversarial', testId],
    () => api.adversarialApi.getStatus(testId!),
    {
      enabled: !!testId,
      refetchInterval: 2000,
    }
  );

  // Start test mutation
  const startMutation = useMutation({
    mutationFn: (data: any) => api.adversarialApi.start(data),
  });

  // Approve patch mutation
  const approveMutation = useMutation({
    mutationFn: ({ testId, round, approved, feedback }: any) =>
      api.adversarialApi.approvePatch(testId, { round, approved, feedback }),
  });

  // Stop test mutation
  const stopMutation = useMutation({
    mutationFn: (id: string) => api.adversarialApi.stop(id),
  });

  return {
    test: statusQuery.data,
    isLoading: statusQuery.isLoading,
    error: statusQuery.error,
    refetch: statusQuery.refetch,
    startTest: startMutation.mutateAsync,
    approvePatch: approveMutation.mutateAsync,
    stopTest: stopMutation.mutateAsync,
    isStarting: startMutation.isPending,
    isApproving: approveMutation.isPending,
    isStopping: stopMutation.isPending,
  };
}

/**
 * Adversarial tests list hook
 */
export function useAdversarialTests(params?: {
  status?: string;
  limit?: number;
  offset?: number;
}) {
  const { useGet } = useApi();

  return useGet(
    ['adversarial-tests', params],
    () => api.adversarialApi.list(params),
    {
      staleTime: 30000,
    }
  );
}

/**
 * Analytics hooks
 */
export function useAnalytics(dateRange?: {
  start: string;
  end: string;
  granularity?: 'hour' | 'day' | 'week' | 'month';
}) {
  const { useGet } = useApi();

  const metricsQuery = useGet(
    ['analytics', 'metrics', dateRange],
    () => api.analytics.getMetrics(dateRange as any),
    {
      enabled: !!dateRange,
      staleTime: 60000, // 1 minute
    }
  );

  const performanceQuery = useGet(
    ['analytics', 'performance'],
    () => api.analytics.getPerformance(),
    {
      staleTime: 300000, // 5 minutes
    }
  );

  return {
    metrics: metricsQuery.data,
    performance: performanceQuery.data,
    isLoading: metricsQuery.isLoading || performanceQuery.isLoading,
    error: metricsQuery.error || performanceQuery.error,
    refetch: () => {
      metricsQuery.refetch();
      performanceQuery.refetch();
    },
  };
}

/**
 * Content hooks
 */
export function useContent(contentId?: string) {
  const { useGet } = useApi();
  const queryClient = useQueryClient();

  // Get single content
  const contentQuery = useGet(
    ['content', contentId],
    () => api.contentApi.getById(contentId!),
    {
      enabled: !!contentId,
    }
  );

  // Update content mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) =>
      api.contentApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['content'] });
    },
  });

  // Delete content mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.contentApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['content'] });
    },
  });

  return {
    content: contentQuery.data,
    isLoading: contentQuery.isLoading,
    error: contentQuery.error,
    updateContent: updateMutation.mutateAsync,
    deleteContent: deleteMutation.mutateAsync,
    isUpdating: updateMutation.isPending,
    isDeleting: deleteMutation.isPending,
  };
}

/**
 * Content list hook
 */
export function useContentList(params?: {
  tag?: string;
  language?: string;
  limit?: number;
  offset?: number;
}) {
  const { useGet } = useApi();

  return useGet(
    ['content', params],
    () => api.contentApi.list(params),
    {
      staleTime: 60000,
    }
  );
}

/**
 * Knowledge graph hook
 */
export function useKnowledgeGraph() {
  const queryClient = useQueryClient();

  // This would call a dedicated knowledge graph endpoint if available
  // For now, we'll use the content list
  const { data: contents } = useContentList();

  // Transform content into graph data
  const graphData = useCallback(() => {
    if (!Array.isArray(contents?.content)) return { nodes: [], edges: [] };

    const nodes = contents.content.map((item) => ({
      id: item.artifact_id,
      label: item.title,
      type: item.language || 'unknown',
      data: item,
    }));

    const edges: Array<{
      id: string;
      source: string;
      target: string;
      type: string;
    }> = [];

    // Create edges based on shared tags
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const node1 = nodes[i];
        const node2 = nodes[j];
        const sharedTags = (node1.data.tags || []).filter((tag: string) =>
          (node2.data.tags || []).includes(tag)
        );

        if (sharedTags.length > 0) {
          edges.push({
            id: `${node1.id}-${node2.id}`,
            source: node1.id,
            target: node2.id,
            type: 'shared_tag',
          });
        }
      }
    }

    return { nodes, edges };
  }, [contents]);

  return {
    graphData: graphData(),
    isLoading: !contents,
  };
}

/**
 * Monitoring hooks
 */
export function useMonitoring(refreshInterval = 5000) {
  const { useGet } = useApi();

  const healthQuery = useGet(
    ['monitoring', 'health'],
    () => api.monitoring.getHealth(),
    {
      refetchInterval,
    }
  );

  return {
    health: healthQuery.data,
    isLoading: healthQuery.isLoading,
    error: healthQuery.error,
  };
}

/**
 * Configuration hooks
 */
export function useConfig() {
  const { useGet } = useApi();
  const queryClient = useQueryClient();
  const updateGlobalSettings = useSettingsStore((state) => state.updateGlobalSettings);

  const providersQuery = useGet(
    ['config', 'providers'],
    () => api.config.getProviders(),
    {
      staleTime: 600000, // 10 minutes
    }
  );

  const parametersQuery = useGet(
    ['config', 'parameters'],
    () => api.config.getParameters(),
    {
      staleTime: 300000, // 5 minutes
    }
  );

  useEffect(() => {
    const params = parametersQuery.data;
    if (!params) {
      return;
    }

    const generation: Partial<{
      temperature: number;
      top_p: number;
      max_tokens: number;
    }> = {};
    const evolution: Partial<{
      max_iterations: number;
      population_size: number;
    }> = {};

    if (params.generation?.temperature !== undefined) {
      generation.temperature = params.generation.temperature;
    }
    if (params.generation?.top_p !== undefined) {
      generation.top_p = params.generation.top_p;
    }
    if (params.generation?.max_tokens !== undefined) {
      generation.max_tokens = params.generation.max_tokens;
    }

    if (params.evolution?.max_iterations !== undefined) {
      evolution.max_iterations = params.evolution.max_iterations;
    }
    if (params.evolution?.population_size !== undefined) {
      evolution.population_size = params.evolution.population_size;
    }

    if (Object.keys(generation).length || Object.keys(evolution).length) {
      updateGlobalSettings({ generation, evolution });
    }
  }, [parametersQuery.data, updateGlobalSettings]);

  // Save API key mutation
  const saveApiKeyMutation = useMutation({
    mutationFn: ({ provider, apiKey }: { provider: string; apiKey: string }) =>
      api.config.saveApiKey(provider, apiKey),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  // Update parameters mutation
  const updateParametersMutation = useMutation({
    mutationFn: (params: any) => api.config.updateParameters(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config', 'parameters'] });
    },
  });

  return {
    providers: providersQuery.data,
    parameters: parametersQuery.data,
    isLoading: providersQuery.isLoading || parametersQuery.isLoading,
    saveApiKey: saveApiKeyMutation.mutateAsync,
    updateParameters: updateParametersMutation.mutateAsync,
    isSavingKey: saveApiKeyMutation.isPending,
    isUpdatingParams: updateParametersMutation.isPending,
  };
}

/**
 * LeanAide hooks
 */
export function useLeanAide() {
  const queryClient = useQueryClient();
  const { useGet } = useApi();

  // Generate proof mutation
  const generateProofMutation = useMutation({
    mutationFn: (data: any) => api.leanaide.generateProof(data),
  });

  // Verify proof mutation
  const verifyProofMutation = useMutation({
    mutationFn: (code: string) => api.leanaide.verifyProof(code),
  });

  // Get models
  const modelsQuery = useGet(
    ['leanaide', 'models'],
    () => api.leanaide.getModels(),
    {
      staleTime: 600000,
    }
  );

  return {
    models: modelsQuery.data,
    generateProof: generateProofMutation.mutateAsync,
    verifyProof: verifyProofMutation.mutateAsync,
    isGenerating: generateProofMutation.isPending,
    isVerifying: verifyProofMutation.isPending,
  };
}
