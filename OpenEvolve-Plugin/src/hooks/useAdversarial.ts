import errorLogger from '@/utils/errorLogging';
// @ts-nocheck
import { useState, useCallback, useEffect, useRef } from 'react';
import { adversarialApi } from '../services/api/endpoints';
import { createAdversarialWebSocket, WebSocketClient } from '../services/api/websocket';
import { useEvolutionStore } from '../stores/evolutionStore';
import { gracefulErrorHandler } from '../utils/gracefulErrorHandler';
import type { AdversarialTest } from '../stores/evolutionStore';

/**
 * Adversarial testing parameters
 */
export interface AdversarialParams {
  content: string;
  attack_modes: string[];
  parameters: {
    num_rounds: number;
    red_team_models: Array<{ provider: string; model: string }>;
    blue_team_models: Array<{ provider: string; model: string }>;
  };
}

/**
 * Adversarial state
 */
export interface AdversarialState {
  data: AdversarialTest | null;
  loading: boolean;
  error: Error | null;
  progress: number;
  currentRound: number;
  totalRounds: number;
}

/**
 * Custom hook for adversarial testing
 * Manages red team vs blue team security testing workflows
 */
export function useAdversarial(testId?: string) {
  const [state, setState] = useState<AdversarialState>({
    data: null,
    loading: false,
    error: null,
    progress: 0,
    currentRound: 0,
    totalRounds: 0,
  });

  const wsClientRef = useRef<WebSocketClient | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [isWsConnected, setIsWsConnected] = useState(false);

  const {
    adversarialTest,
    setAdversarialTest,
    updateAdversarialTest,
    approvePatch,
    setIsAdversarialRunning,
    setLoading,
    setError,
  } = useEvolutionStore();

  /**
   * Execute adversarial test
   */
  const execute = useCallback(async (params: AdversarialParams): Promise<void> => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setState(prev => ({
        ...prev,
        loading: true,
        error: null,
        progress: 0,
        currentRound: 0,
        totalRounds: params.parameters.num_rounds,
      }));
      setLoading(true);
      setError(null);

      abortControllerRef.current = new AbortController();

      const response = await adversarialApi.start(params);

      const testRecord = {
        test_id: response.test_id,
        status: response.status as any,
        current_round: 0,
        total_rounds: params.parameters.num_rounds,
        red_team_results: [],
        blue_team_results: [],
        vulnerabilities_found: 0,
        patches_generated: 0,
        patches_approved: 0,
        created_at: response.created_at,
        websocket_url: response.websocket_url,
      };
      setAdversarialTest(testRecord);

      setIsAdversarialRunning(true);
      setState(prev => ({ ...prev, data: testRecord as any }));

      // Setup WebSocket for real-time updates
      if (response.websocket_url) {
        wsClientRef.current = createAdversarialWebSocket(
          response.test_id,
          {
            onMessage: (message) => {
              if (message.type === 'progress_update') {
                const { round, total_rounds } = message.data;
                const progress = total_rounds ? (round / total_rounds) * 100 : 0;
                setState(prev => ({
                  ...prev,
                  progress,
                  currentRound: round,
                }));
              } else if (message.type === 'attack_generated') {
                // Red team generated an attack
                const { round, attack_mode, payload, success } = message.data;
                updateAdversarialTest(response.test_id, {
                  current_round: round,
                  red_team_results: [
                    {
                      round,
                      attack_mode,
                      success,
                      payload,
                    },
                  ],
                });
              } else if (message.type === 'patch_generated') {
                // Blue team generated a patch
                const { round, vulnerability, patch } = message.data;
                updateAdversarialTest(response.test_id, {
                  blue_team_results: [
                    {
                      round,
                      vulnerability,
                      patch,
                      patch_approved: false,
                    },
                  ],
                  patches_generated: (state.data?.patches_generated || 0) + 1,
                });
              } else if (message.type === 'patch_approved') {
                // Patch was approved
                const { round, approved } = message.data;
                approvePatch(response.test_id, round, approved);
                if (approved) {
                  updateAdversarialTest(response.test_id, {
                    patches_approved: (state.data?.patches_approved || 0) + 1,
                  });
                }
              } else if (message.type === 'test_complete') {
                setState(prev => ({ ...prev, loading: false, progress: 100 }));
                setLoading(false);
                setIsAdversarialRunning(false);
              } else if (message.type === 'error') {
                const error = new Error(message.data.message || 'Adversarial test error');
                setState(prev => ({ ...prev, error, loading: false }));
                setError(error.message);
                setLoading(false);
                setIsAdversarialRunning(false);
              }
            },
            onConnectionChange: (connState) => {
              // Connection state change - no logging needed
              setIsWsConnected(connState === 'connected');
            },
            onError: (error) => {
              errorLogger.logError(error, 'error', {
                component: 'useAdversarial',
                function: 'onError',
                additionalData: { error }
              });
            },
          }
        );

        wsClientRef.current.connect();
      }

      return response;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useAdversarial',
        function: 'execute',
        operation: 'EXECUTE_ADVERSARIAL_TEST',
        additionalData: {
          attackModes: params.attack_modes,
          numRounds: params.parameters.num_rounds
        }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      setIsAdversarialRunning(false);
    }
  }, [setAdversarialTest, setIsAdversarialRunning, setLoading, setError, adversarialTest, state.data, updateAdversarialTest, approvePatch]);

  /**
   * Get current test status
   */
  const getStatus = useCallback(async (): Promise<AdversarialTest | null> => {
    if (!testId) {
      return null;
    }

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      const status = await adversarialApi.getStatus(testId);
      setAdversarialTest(status);
      setState(prev => ({ ...prev, data: status }));
      return status;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'useAdversarial',
        function: 'getStatus',
        operation: 'GET_ADVERSARIAL_STATUS',
        additionalData: { testId }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error }));
      setError(error.message);
      return null;
    }

    return result.data!;
  }, [testId, setAdversarialTest, setError]);

  /**
   * Get test results
   */
  const getResults = useCallback((): AdversarialTest | null => {
    return state.data || adversarialTest;
  }, [state.data, adversarialTest]);

  /**
   * Cancel test
   */
  const cancel = useCallback(async (): Promise<void> => {
    if (!testId) {
      return;
    }

    setState(prev => ({ ...prev, loading: true }));
    setLoading(true);

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      // Cancel any pending requests
      abortControllerRef.current?.abort();

      // Stop test via API
      await adversarialApi.stop(testId);

      // Disconnect WebSocket
      wsClientRef.current?.disconnect();

      setIsAdversarialRunning(false);
      setLoading(false);
      setState(prev => ({ ...prev, loading: false, progress: 0 }));
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useAdversarial',
        function: 'cancel',
        operation: 'CANCEL_ADVERSARIAL_TEST',
        additionalData: { testId }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
    }
  }, [testId, setIsAdversarialRunning, setLoading, setError]);

  /**
   * Approve or reject a patch
   */
  const approveOrRejectPatch = useCallback(async (
    round: number,
    approved: boolean,
    feedback?: string
  ): Promise<void> => {
    if (!testId) {
      return;
    }

    setState(prev => ({ ...prev, loading: true }));
    setLoading(true);

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      await adversarialApi.approvePatch(testId, { round, approved, feedback });
      approvePatch(testId, round, approved, feedback);

      if (approved) {
        updateAdversarialTest(testId, {
          patches_approved: (state.data?.patches_approved || 0) + 1,
        });
      }

      setLoading(false);
      setState(prev => ({ ...prev, loading: false }));
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useAdversarial',
        function: 'approveOrRejectPatch',
        operation: 'APPROVE_REJECT_PATCH',
        additionalData: { testId, round, approved }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
    }
  }, [testId, state.data, approvePatch, updateAdversarialTest, setLoading, setError]);

  /**
   * Reset state
   */
  const reset = useCallback((): void => {
    setState({
      data: null,
      loading: false,
      error: null,
      progress: 0,
      currentRound: 0,
      totalRounds: 0,
    });
    setLoading(false);
    setError(null);
  }, [setLoading, setError]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      wsClientRef.current?.disconnect();
      abortControllerRef.current?.abort();
    };
  }, []);

  /**
   * Poll for status updates if not using WebSocket
   */
  useEffect(() => {
    if (!testId || isWsConnected) {
      return;
    }

    const interval = setInterval(() => {
      getStatus();
    }, 2000);

    return () => clearInterval(interval);
  }, [testId, getStatus, isWsConnected]);

  return {
    ...state,
    execute,
    getStatus,
    getResults,
    cancel,
    approvePatch: approveOrRejectPatch,
    reset,
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
  const [state, setState] = useState<{
    data: AdversarialTest[] | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const { setAdversarialTests } = useEvolutionStore();

  const fetchTests = useCallback(async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setState(prev => ({ ...prev, loading: true, error: null }));

      const response = await adversarialApi.list(params);
      const tests = response?.tests || [];
      setAdversarialTests(tests);
      setState(prev => ({ ...prev, data: tests, loading: false }));

      return tests;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'useAdversarialTests',
        function: 'fetchTests',
        operation: 'FETCH_ADVERSARIAL_TESTS',
        additionalData: { params }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, [params, setAdversarialTests]);

  useEffect(() => {
    fetchTests();
  }, [fetchTests]);

  return {
    ...state,
    refetch: fetchTests,
  };
}
