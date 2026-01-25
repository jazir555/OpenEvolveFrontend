import { useState, useCallback, useEffect, useRef } from 'react';
import { evolutionApi } from '@/services/api/endpoints';
import { createEvolutionWebSocket, WebSocketClient } from '@/services/api/websocket';
import { useEvolutionStore } from '@/stores/evolutionStore';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import errorLogger from '@/utils/errorLogging';
import type { WorkflowExecution } from '@/stores';

/**
 * Evolution parameters
 */
export interface EvolutionParams {
  content: string;
  mode: 'standard' | 'quality_diversity' | 'island_model';
  parameters: {
    max_iterations: number;
    population_size: number;
    temperature: number;
    top_p: number;
  };
  models: Array<{
    provider: string;
    model: string;
    api_key: string;
  }>;
}

/**
 * Evolution state
 */
export interface EvolutionState {
  data: WorkflowExecution | null;
  loading: boolean;
  error: Error | null;
  progress: number;
}

/**
 * Custom hook for genetic algorithm evolution
 * Manages evolutionary optimization workflows with real-time updates
 */
export function useEvolution(evolutionId?: string) {
  const [state, setState] = useState<EvolutionState>({
    data: null,
    loading: false,
    error: null,
    progress: 0,
  });

  const wsClientRef = useRef<WebSocketClient | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [isWsConnected, setIsWsConnected] = useState(false);

  const {
    evolution,
    setEvolution,
    updateEvolution,
    setIsEvolutionRunning,
    setLoading,
    setError,
  } = useEvolutionStore();

  /**
   * Execute evolution
   */
  const execute = useCallback(async (params: EvolutionParams): Promise<void> => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setState(prev => ({ ...prev, loading: true, error: null, progress: 0 }));
      setLoading(true);
      setError(null);

      abortControllerRef.current = new AbortController();

      const response = await evolutionApi.start(params);

      const evolutionRecord = {
        evolution_id: response.evolution_id,
        status: response.status as any,
        created_at: response.created_at,
        content: params.content,
        mode: params.mode,
        parameters: params.parameters,
        iterations: [],
        current_iteration: 0,
        websocket_url: response.websocket_url,
      };
      setEvolution(evolutionRecord);

      setIsEvolutionRunning(true);
      setState(prev => ({ ...prev, data: evolutionRecord as any }));

      // Setup WebSocket for real-time updates
      if (response.websocket_url) {
        try {
          wsClientRef.current = createEvolutionWebSocket(
            response.evolution_id,
            {
              onMessage: (message) => {
                try {
                  if (message.type === 'progress_update') {
                    const progress = message.data.progress || 0;
                    setState(prev => ({ ...prev, progress }));
                  } else if (message.type === 'evolution_complete') {
                    setState(prev => ({ ...prev, loading: false, progress: 100 }));
                    setLoading(false);
                    setIsEvolutionRunning(false);
                  } else if (message.type === 'error') {
                    const error = new Error(message.data.message || 'Evolution error');
                    setState(prev => ({ ...prev, error, loading: false }));
                    setError(error.message);
                    setLoading(false);
                    setIsEvolutionRunning(false);
                  }
                } catch (msgError) {
                  errorLogger.logError('Error processing WebSocket message', 'error', {
                    component: 'useEvolution',
                    function: 'onMessage',
                    additionalData: { error: msgError }
                  });
                }
              },
              onConnectionChange: (connState) => {
                try {
                  // Connection state change - no logging needed for normal operation
                  setIsWsConnected(connState === 'connected');
                } catch (connError) {
                  errorLogger.logError('Error handling connection change', 'error', {
                    component: 'useEvolution',
                    function: 'onConnectionChange',
                    additionalData: { error: connError, state: connState }
                  });
                }
              },
              onError: (error) => {
                try {
                  errorLogger.logError('WebSocket error', 'error', {
                    component: 'useEvolution',
                    function: 'onError',
                    additionalData: { error }
                  });
                } catch (wsError) {
                  errorLogger.logError('Error logging WebSocket error', 'error', {
                    component: 'useEvolution',
                    function: 'onError',
                    additionalData: { originalError: error, loggingError: wsError }
                  });
                }
              },
            }
          );

          wsClientRef.current.connect();
        } catch (wsError) {
          errorLogger.logError('Failed to setup WebSocket', 'error', {
            component: 'useEvolution',
            function: 'runEvolution',
            additionalData: { error: wsError }
          });
          // Continue execution without WebSocket
        }
      }

      return response;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useEvolution',
        function: 'execute',
        operation: 'EXECUTE_EVOLUTION',
        additionalData: { mode: params.mode, parameters: params.parameters }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      setIsEvolutionRunning(false);
    }
  }, [setEvolution, setIsEvolutionRunning, setLoading, setError, evolution]);

  /**
   * Get current evolution status
   */
  const getStatus = useCallback(async (): Promise<WorkflowExecution | null> => {
    if (!evolutionId) {
      return null;
    }

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      const status = await evolutionApi.getStatus(evolutionId);
      setEvolution(status);
      setState(prev => ({ ...prev, data: status }));
      return status;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'useEvolution',
        function: 'getStatus',
        operation: 'GET_EVOLUTION_STATUS',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error }));
      setError(error.message);
      return null;
    }

    return result.data!;
  }, [evolutionId, setEvolution, setError]);

  /**
   * Get evolution results
   */
  const getResults = useCallback((): WorkflowExecution | null => {
    try {
      return state.data || evolution;
    } catch (error) {
      errorLogger.logError('Error getting evolution results', 'error', {
        component: 'useEvolution',
        function: 'getResults',
        additionalData: { error }
      });
      return null;
    }
  }, [state.data, evolution]);

  /**
   * Cancel evolution
   */
  const cancel = useCallback(async (): Promise<void> => {
    if (!evolutionId) {
      return;
    }

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setState(prev => ({ ...prev, loading: true }));
      setLoading(true);

      // Cancel any pending requests
      abortControllerRef.current?.abort();

      // Stop evolution via API
      await evolutionApi.stop(evolutionId);

      // Disconnect WebSocket
      if (wsClientRef.current) {
        try {
          wsClientRef.current.disconnect();
        } catch (disconnectError) {
          errorLogger.logError('Error disconnecting WebSocket', 'error', {
            component: 'useEvolution',
            function: 'disconnect',
            additionalData: { error: disconnectError }
          });
        }
      }

      setIsEvolutionRunning(false);
      setLoading(false);
      setState(prev => ({ ...prev, loading: false, progress: 0 }));
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useEvolution',
        function: 'cancel',
        operation: 'CANCEL_EVOLUTION',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
    }
  }, [evolutionId, setIsEvolutionRunning, setLoading, setError]);

  /**
   * Pause evolution
   */
  const pause = useCallback(async (): Promise<void> => {
    if (!evolutionId) {
      return;
    }

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      await evolutionApi.pause(evolutionId);
      updateEvolution(evolutionId, { status: 'paused' as any });
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useEvolution',
        function: 'pause',
        operation: 'PAUSE_EVOLUTION',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error }));
      setError(error.message);
    }
  }, [evolutionId, updateEvolution, setError]);

  /**
   * Resume evolution
   */
  const resume = useCallback(async (): Promise<void> => {
    if (!evolutionId) {
      return;
    }

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      await evolutionApi.resume(evolutionId);
      updateEvolution(evolutionId, { status: 'running' as any });
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'useEvolution',
        function: 'resume',
        operation: 'RESUME_EVOLUTION',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error }));
      setError(error.message);
    }
  }, [evolutionId, updateEvolution, setError]);

  /**
   * Reset state
   */
  const reset = useCallback((): void => {
    try {
      setState({
        data: null,
        loading: false,
        error: null,
        progress: 0,
      });
      setLoading(false);
      setError(null);
    } catch (error) {
      errorLogger.logError('Error resetting evolution state', 'error', {
        component: 'useEvolution',
        function: 'reset',
        additionalData: { error }
      });
    }
  }, [setLoading, setError]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      try {
        if (wsClientRef.current) {
          wsClientRef.current.disconnect();
        }
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
      } catch (cleanupError) {
        errorLogger.logError('Error during cleanup', 'error', {
          component: 'useEvolution',
          function: 'reset',
          additionalData: { error: cleanupError }
        });
      }
    };
  }, []);

  /**
   * Poll for status updates if not using WebSocket
   */
  useEffect(() => {
    if (!evolutionId || isWsConnected) {
      return;
    }

    let intervalId: NodeJS.Timeout;
    try {
      intervalId = setInterval(() => {
        try {
          getStatus();
        } catch (intervalError) {
          errorLogger.logError('Error in status polling interval', 'error', {
            component: 'useEvolution',
            function: 'startPolling',
            additionalData: { error: intervalError }
          });
        }
      }, 2000);
    } catch (setupError) {
      errorLogger.logError('Error setting up polling interval', 'error', {
        component: 'useEvolution',
        function: 'startPolling',
        additionalData: { error: setupError }
      });
    }

    return () => {
      try {
        if (intervalId) {
          clearInterval(intervalId);
        }
      } catch (clearError) {
        errorLogger.logError('Error clearing interval', 'error', {
          component: 'useEvolution',
          function: 'stopPolling',
          additionalData: { error: clearError }
        });
      }
    };
  }, [evolutionId, getStatus, isWsConnected]);

  return {
    ...state,
    execute,
    getStatus,
    getResults,
    cancel,
    pause,
    resume,
    reset,
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
  const [state, setState] = useState<{
    data: WorkflowExecution[] | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const { setEvolutions } = useEvolutionStore();

  const fetchEvolutions = useCallback(async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setState(prev => ({ ...prev, loading: true, error: null }));

      const response = await evolutionApi.list(params);
      const evolutions = response?.evolutions || [];
      setEvolutions(evolutions);
      setState(prev => ({ ...prev, data: evolutions, loading: false }));

      return evolutions;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: false,
      logError: true,
      context: {
        component: 'useEvolutions',
        function: 'fetchEvolutions',
        operation: 'FETCH_EVOLUTIONS',
        additionalData: { params }
      }
    });

    if (!result.success) {
      const error = result.error instanceof Error ? result.error : new Error(String(result.error));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, [params, setEvolutions]);

  useEffect(() => {
    try {
      fetchEvolutions();
    } catch (effectError) {
      errorLogger.logError('Error in fetchEvolutions effect', 'error', {
        component: 'useEvolution',
        function: 'fetchEvolutions',
        additionalData: { error: effectError }
      });
    }
  }, [fetchEvolutions]);

  return {
    ...state,
    refetch: fetchEvolutions,
  };
}
