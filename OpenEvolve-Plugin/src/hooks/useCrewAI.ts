import errorLogger from '@/utils/errorLogging';
import { useState, useCallback, useEffect, useRef } from 'react';
import { apiClient } from '../services/api/client';

/**
 * Code generation parameters
 */
export interface CodeGenerationParams {
  requirement: string;
  language: string;
  framework?: string;
  context?: string;
  include_tests?: boolean;
  include_comments?: boolean;
  style_guide?: string;
}

/**
 * Generated code result
 */
export interface GeneratedCode {
  code_id: string;
  language: string;
  code: string;
  tests?: string;
  documentation?: string;
  metadata: {
    generated_at: string;
    model_used: string;
    confidence_score: number;
    tokens_used: number;
  };
}

/**
 * Code review result
 */
export interface CodeReviewResult {
  review_id: string;
  code_id: string;
  issues: Array<{
    severity: 'low' | 'medium' | 'high' | 'critical';
    category: string;
    description: string;
    suggestion?: string;
    line_number?: number;
  }>;
  metrics: {
    complexity: number;
    maintainability: number;
    test_coverage?: number;
    documentation_coverage: number;
  };
  overall_score: number;
}

/**
 * Code optimization result
 */
export interface CodeOptimizationResult {
  optimized_code: string;
  improvements: Array<{
    type: string;
    description: string;
    impact: 'low' | 'medium' | 'high';
  }>;
  performance_gain: number;
}

/**
 * CrewAI state
 */
export interface CrewAIState {
  data: GeneratedCode | CodeReviewResult | CodeOptimizationResult | null;
  loading: boolean;
  error: Error | null;
  progress: number;
}

/**
 * Custom hook for CrewAI code generation bridge
 * Manages code generation, review, and optimization workflows
 */
export function useCrewAI(codeId?: string) {
  const [state, setState] = useState<CrewAIState>({
    data: null,
    loading: false,
    error: null,
    progress: 0,
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const operationRef = useRef<'generate' | 'review' | 'optimize' | null>(null);

  /**
   * Execute code generation
   */
  const execute = useCallback(async (
    params: CodeGenerationParams
  ): Promise<GeneratedCode | null> => {
    setState(prev => ({
      ...prev,
      loading: true,
      error: null,
      progress: 0,
    }));
    operationRef.current = 'generate';

    let progressInterval: ReturnType<typeof setInterval> | null = null;
    try {
      abortControllerRef.current = new AbortController();

      // Simulate progress
      progressInterval = setInterval(() => {
        setState(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 10, 90),
        }));
      }, 300);

      const response = await apiClient.post<GeneratedCode>(
        '/crewai/generate',
        params
      );

      setState(prev => ({
        ...prev,
        data: response,
        loading: false,
        progress: 100,
      }));

      operationRef.current = null;
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({
        ...prev,
        error,
        loading: false,
        progress: 0,
      }));
      operationRef.current = null;
      return null;
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    }
  }, []);

  /**
   * Review code
   */
  const review = useCallback(async (
    code: string,
    language: string
  ): Promise<CodeReviewResult | null> => {
    setState(prev => ({
      ...prev,
      loading: true,
      error: null,
      progress: 0,
    }));
    operationRef.current = 'review';

    let progressInterval: ReturnType<typeof setInterval> | null = null;
    try {
      abortControllerRef.current = new AbortController();

      progressInterval = setInterval(() => {
        setState(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 15, 90),
        }));
      }, 300);

      const response = await apiClient.post<CodeReviewResult>(
        '/crewai/review',
        { code, language }
      );

      setState(prev => ({
        ...prev,
        data: response,
        loading: false,
        progress: 100,
      }));

      operationRef.current = null;
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({
        ...prev,
        error,
        loading: false,
        progress: 0,
      }));
      operationRef.current = null;
      return null;
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    }
  }, []);

  /**
   * Optimize code
   */
  const optimize = useCallback(async (
    code: string,
    language: string,
    optimization_goals?: string[]
  ): Promise<CodeOptimizationResult | null> => {
    setState(prev => ({
      ...prev,
      loading: true,
      error: null,
      progress: 0,
    }));
    operationRef.current = 'optimize';

    let progressInterval: ReturnType<typeof setInterval> | null = null;
    try {
      abortControllerRef.current = new AbortController();

      progressInterval = setInterval(() => {
        setState(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 12, 90),
        }));
      }, 300);

      const response = await apiClient.post<CodeOptimizationResult>(
        '/crewai/optimize',
        { code, language, optimization_goals: optimization_goals || ['performance', 'readability'] }
      );

      setState(prev => ({
        ...prev,
        data: response,
        loading: false,
        progress: 100,
      }));

      operationRef.current = null;
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({
        ...prev,
        error,
        loading: false,
        progress: 0,
      }));
      operationRef.current = null;
      return null;
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    }
  }, []);

  /**
   * Get status of code operation
   */
  const getStatus = useCallback(async (): Promise<any> => {
    if (!codeId) {
      return null;
    }

    try {
      const response = await apiClient.get<any>(`/crewai/status/${codeId}`);
      setState(prev => ({ ...prev, data: response }));
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error }));
      return null;
    }
  }, [codeId]);

  /**
   * Get results
   */
  const getResults = useCallback((): (GeneratedCode | CodeReviewResult | CodeOptimizationResult | null) => {
    return state.data;
  }, [state.data]);

  /**
   * Cancel operation
   */
  const cancel = useCallback((): void => {
    abortControllerRef.current?.abort();
    setState(prev => ({
      ...prev,
      loading: false,
      progress: 0,
    }));
    operationRef.current = null;
  }, []);

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
  }, []);

  /**
   * Get supported languages
   */
  const getSupportedLanguages = useCallback(async (): Promise<string[]> => {
    try {
      const response = await apiClient.get<{ languages: string[] }>('/crewai/languages');
      return response.languages;
    } catch (err) {
      errorLogger.logError(err, 'error', { component: 'useCrewAI.ts', function: 'unknown', additionalData: { operation: 'Failed to fetch supported languages:' } });
      return [];
    }
  }, []);

  /**
   * Get code templates
   */
  const getTemplates = useCallback(async (
    language: string
  ): Promise<Array<{ name: string; description: string; template: string }>> => {
    try {
      const response = await apiClient.get<{ templates: any[] }>(
        `/crewai/templates/${language}`
      );
      return response.templates;
    } catch (err) {
      errorLogger.logError(err, 'error', { component: 'useCrewAI.ts', function: 'unknown', additionalData: { operation: 'Failed to fetch templates:' } });
      return [];
    }
  }, []);

  /**
   * Apply fix from review
   */
  const applyFix = useCallback(async (
    code: string,
    fixIndex: number
  ): Promise<string | null> => {
    try {
      const response = await apiClient.post<{ fixed_code: string }>(
        '/crewai/apply-fix',
        { code, fix_index: fixIndex }
      );
      return response.fixed_code;
    } catch (err) {
      errorLogger.logError(err, 'error', { component: 'useCrewAI.ts', function: 'unknown', additionalData: { operation: 'Failed to apply fix:' } });
      return null;
    }
  }, []);

  /**
   * Get code metrics
   */
  const getCodeMetrics = useCallback(async (
    code: string,
    language: string
  ): Promise<{
    lines_of_code: number;
    complexity: number;
    maintainability_index: number;
    technical_debt: number;
  } | null> => {
    try {
      const response = await apiClient.post<any>('/crewai/metrics', { code, language });
      return response;
    } catch (err) {
      errorLogger.logError(err, 'error', { component: 'useCrewAI.ts', function: 'unknown', additionalData: { operation: 'Failed to get code metrics:' } });
      return null;
    }
  }, []);

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
    currentOperation: operationRef.current,
    execute,
    review,
    optimize,
    getStatus,
    getResults,
    cancel,
    reset,
    getSupportedLanguages,
    getTemplates,
    applyFix,
    getCodeMetrics,
  };
}

/**
 * CrewAI templates hook
 */
export function useCrewAITemplates() {
  const [state, setState] = useState<{
    data: Array<{
      id: string;
      name: string;
      language: string;
      category: string;
      description: string;
      template: string;
    }>;
    loading: boolean;
    error: Error | null;
  }>({
    data: [],
    loading: false,
    error: null,
  });

  const fetchTemplates = useCallback(async (language?: string, category?: string) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const params = new URLSearchParams();
      if (language) params.append('language', language);
      if (category) params.append('category', category);

      const response = await apiClient.get<{ templates: any[] }>(
        `/crewai/templates?${params.toString()}`
      );

      setState(prev => ({
        ...prev,
        data: response.templates,
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

/**
 * CrewAI code history hook
 */
export function useCrewAIHistory() {
  const [state, setState] = useState<{
    data: Array<{
      code_id: string;
      requirement: string;
      language: string;
      generated_at: string;
    }>;
    loading: boolean;
    error: Error | null;
  }>({
    data: [],
    loading: false,
    error: null,
  });

  const fetchHistory = useCallback(async (limit: number = 20) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await apiClient.get<{ history: any[] }>(
        `/crewai/history?limit=${limit}`
      );

      setState(prev => ({
        ...prev,
        data: response.history,
        loading: false,
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return {
    ...state,
    refetch: fetchHistory,
  };
}
