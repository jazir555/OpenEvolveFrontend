import { useState, useCallback, useEffect, useRef } from 'react';
import { leanaideApi } from '../services/api/endpoints';
import { useLeanAideStore } from '../stores/leanaideStore';
import type { LeanCodeOutput, VerificationResult, ProofStatus } from '../stores/leanaideStore';

/**
 * Lean 4 proof generation parameters
 */
export interface LeanProofParams {
  theorem: string;
  proof_attempt?: string;
  model: string;
  temperature: number;
}

/**
 * Proof verification params
 */
export interface VerificationParams {
  code: string;
  timeout?: number;
}

/**
 * LeanAIDE state
 */
export interface LeanAIDEState {
  data: {
    generatedProof: LeanCodeOutput | null;
    verificationResult: VerificationResult | null;
  } | null;
  loading: boolean;
  error: Error | null;
  progress: number;
  status: ProofStatus;
}

/**
 * Custom hook for Lean 4 formal verification
 * Manages theorem proving and proof verification workflows
 */
export function useLeanAIDE() {
  const [state, setState] = useState<LeanAIDEState>({
    data: {
      generatedProof: null,
      verificationResult: null,
    },
    loading: false,
    error: null,
    progress: 0,
    status: 'idle',
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const {
    theorem,
    proofAttempt,
    generatedProof,
    verificationResult,
    modelConfig,
    status,
    setStatus,
    setGeneratedProof,
    setVerificationResult,
    setLoading,
    setError,
    clearOutputs,
  } = useLeanAideStore();

  /**
   * Execute proof generation
   */
  const execute = useCallback(async (params: LeanProofParams): Promise<LeanCodeOutput | null> => {
    setState(prev => ({
      ...prev,
      loading: true,
      error: null,
      progress: 0,
      status: 'generating',
    }));
    setStatus('generating');
    setLoading(true);
    setError(null);

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

      const response = await leanaideApi.generateProof(params);

      setGeneratedProof(response);
      setState(prev => ({
        ...prev,
        data: {
          generatedProof: response,
          verificationResult: prev.data?.verificationResult || null,
        },
        loading: false,
        progress: 100,
        status: 'completed',
      }));

      setStatus('completed');
      setLoading(false);

      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({
        ...prev,
        error,
        loading: false,
        progress: 0,
        status: 'failed',
      }));
      setError(error.message);
      setStatus('failed');
      setLoading(false);
      return null;
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    }
  }, [setStatus, setGeneratedProof, setLoading, setError]);

  /**
   * Verify proof
   */
  const verify = useCallback(async (
    params: VerificationParams
  ): Promise<VerificationResult | null> => {
    setState(prev => ({
      ...prev,
      loading: true,
      error: null,
      status: 'verifying',
    }));
    setStatus('verifying');
    setLoading(true);
    setError(null);

    try {
      abortControllerRef.current = new AbortController();

      const response = await leanaideApi.verifyProof(params.code);

      setVerificationResult(response);
      setState(prev => ({
        ...prev,
        data: {
          generatedProof: prev.data?.generatedProof || null,
          verificationResult: response,
        },
        loading: false,
        status: response.success ? 'completed' : 'failed',
      }));

      setStatus(response.success ? 'completed' : 'failed');
      setLoading(false);

      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({
        ...prev,
        error,
        loading: false,
        status: 'failed',
      }));
      setError(error.message);
      setStatus('failed');
      setLoading(false);
      return null;
    }
  }, [setStatus, setVerificationResult, setLoading, setError]);

  /**
   * Get available models
   */
  const getModels = useCallback(async (): Promise<Array<{ provider: string; models: string[] }>> => {
    try {
      const response = await leanaideApi.getModels();
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      return [];
    }
  }, [setError]);

  /**
   * Get generation status
   */
  const getStatus = useCallback((): ProofStatus => {
    return state.status;
  }, [state.status]);

  /**
   * Get results
   */
  const getResults = useCallback((): {
    generatedProof: LeanCodeOutput | null;
    verificationResult: VerificationResult | null;
  } => {
    return {
      generatedProof: state.data?.generatedProof || generatedProof,
      verificationResult: state.data?.verificationResult || verificationResult,
    };
  }, [state.data, generatedProof, verificationResult]);

  /**
   * Cancel operation
   */
  const cancel = useCallback((): void => {
    abortControllerRef.current?.abort();
    setState(prev => ({
      ...prev,
      loading: false,
      progress: 0,
      status: 'idle',
    }));
    setStatus('idle');
    setLoading(false);
  }, [setStatus, setLoading]);

  /**
   * Reset state
   */
  const reset = useCallback((): void => {
    setState({
      data: {
        generatedProof: null,
        verificationResult: null,
      },
      loading: false,
      error: null,
      progress: 0,
      status: 'idle',
    });
    clearOutputs();
    setStatus('idle');
    setError(null);
  }, [clearOutputs, setStatus, setError]);

  /**
   * Update model configuration
   */
  const updateModelConfig = useCallback((config: Partial<typeof modelConfig>): void => {
    // Store-level update would go here
    // For now, just update local state
    setState(prev => ({ ...prev }));
  }, [modelConfig]);

  /**
   * Run benchmark
   */
  const runBenchmark = useCallback(async (
    dataset: any[],
    model: string,
    evaluator: string
  ): Promise<string | null> => {
    setState(prev => ({ ...prev, loading: true }));
    setLoading(true);

    try {
      const response = await leanaideApi.runBenchmark({
        dataset,
        model,
        evaluator,
      });

      setState(prev => ({ ...prev, loading: false }));
      setLoading(false);

      return response.benchmark_id;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Get benchmark results
   */
  const getBenchmarkResults = useCallback(async (
    benchmarkId: string
  ): Promise<any[] | null> => {
    try {
      const results = await leanaideApi.getBenchmarkResults(benchmarkId);
      return results;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      return null;
    }
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
    theorem,
    proofAttempt,
    modelConfig,
    execute,
    verify,
    getModels,
    getStatus,
    getResults,
    cancel,
    reset,
    updateModelConfig,
    runBenchmark,
    getBenchmarkResults,
  };
}

/**
 * Lean 4 tactics library hook
 */
export function useLeanTactics() {
  const [state, setState] = useState<{
    data: Array<{
      name: string;
      description: string;
      syntax: string;
      example: string;
      category: string;
    }> | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const fetchTactics = useCallback(async (category?: string) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await fetch(`/api/v1/leanaide/tactics${category ? `?category=${category}` : ''}`);
      const tactics = await response.json();

      setState(prev => ({
        ...prev,
        data: tactics.tactics || [],
        loading: false,
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, []);

  useEffect(() => {
    fetchTactics();
  }, [fetchTactics]);

  return {
    ...state,
    refetch: fetchTactics,
  };
}

/**
 * Lean 4 documentation hook
 */
export function useLeanDocs() {
  const [state, setState] = useState<{
    data: {
      library_docs: string;
      tactic_reference: string;
      examples: string;
    } | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const fetchDocs = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await fetch('/api/v1/leanaide/docs');
      const docs = await response.json();

      setState(prev => ({
        ...prev,
        data: docs,
        loading: false,
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  return {
    ...state,
    refetch: fetchDocs,
  };
}
