/**
 * Execution Streaming Hook
 * Handles SSE (Server-Sent Events) for real-time workflow execution updates
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { ExecutionEvent, ExecutionEventType } from '../types/api';

interface UseExecutionStreamOptions {
  onEvent?: (event: ExecutionEvent) => void;
  onError?: (error: Error) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseExecutionStreamReturn {
  connected: boolean;
  error: Error | null;
  reconnectAttempts: number;
  manualReconnect: () => void;
  disconnect: () => void;
}

export function useExecutionStream(
  workflowId: string | null,
  options: UseExecutionStreamOptions = {}
): UseExecutionStreamReturn {
  const {
    onEvent,
    onError,
    onConnected,
    onDisconnected,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options;

  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(true);

  /**
   * Connect to SSE stream
   */
  const connect = useCallback(() => {
    if (!workflowId || !shouldReconnectRef.current) {
      return;
    }

    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
    const url = `${apiBase}/stream/workflow/${workflowId}`;

    console.log(`[SSE] Connecting to ${url}`);

    try {
      const eventSource = new EventSource(url);

      eventSource.onopen = () => {
        console.log('[SSE] Connection opened');
        setConnected(true);
        setError(null);
        setReconnectAttempts(0);
        onConnected?.();
      };

      eventSource.onmessage = (event) => {
        try {
          const data: ExecutionEvent = JSON.parse(event.data);
          console.log('[SSE] Event received:', data);
          onEvent?.(data);
        } catch (err) {
          console.error('[SSE] Failed to parse event:', err);
        }
      };

      eventSource.onerror = (err) => {
        console.error('[SSE] Connection error:', err);
        setConnected(false);

        const error = new Error('SSE connection error');
        setError(error);
        onError?.(error);

        // EventSource will automatically try to reconnect
        // We handle max reconnect attempts manually
      };

      // Listen for specific event types
      eventSource.addEventListener('error', (e) => {
        const event = e as MessageEvent;
        try {
          const data = JSON.parse(event.data);
          console.error('[SSE] Server error event:', data);
          const error = new Error(data.error || 'Server error');
          setError(error);
          onError?.(error);
        } catch (err) {
          console.error('[SSE] Failed to parse error event:', err);
        }
      });

      eventSourceRef.current = eventSource;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to create EventSource');
      setError(error);
      onError?.(error);
    }
  }, [workflowId, onEvent, onError, onConnected]);

  /**
   * Disconnect from SSE stream
   */
  const disconnect = useCallback(() => {
    console.log('[SSE] Disconnecting');
    shouldReconnectRef.current = false;

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    setConnected(false);
    onDisconnected?.();
  }, [onDisconnected]);

  /**
   * Manual reconnect
   */
  const manualReconnect = useCallback(() => {
    console.log('[SSE] Manual reconnect requested');
    shouldReconnectRef.current = true;
    setReconnectAttempts(0);
    connect();
  }, [connect]);

  /**
   * Setup connection and cleanup
   */
  useEffect(() => {
    if (!workflowId) {
      return;
    }

    shouldReconnectRef.current = true;
    connect();

    return () => {
      disconnect();
    };
  }, [workflowId, connect, disconnect]);

  /**
   * Handle reconnection logic
   */
  useEffect(() => {
    if (!connected && reconnectAttempts < maxReconnectAttempts && workflowId) {
      console.log(`[SSE] Reconnecting in ${reconnectInterval}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);

      reconnectTimeoutRef.current = setTimeout(() => {
        setReconnectAttempts((prev) => prev + 1);
        connect();
      }, reconnectInterval);

      return () => {
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
      };
    } else if (!connected && reconnectAttempts >= maxReconnectAttempts) {
      console.error('[SSE] Max reconnection attempts reached');
      const error = new Error('Max reconnection attempts reached');
      setError(error);
      onError?.(error);
    }
  }, [connected, reconnectAttempts, maxReconnectAttempts, reconnectInterval, workflowId, connect, onError]);

  return {
    connected,
    error,
    reconnectAttempts,
    manualReconnect,
    disconnect,
  };
}

// ============================================================================
// Event Type Specific Hooks
// ============================================================================

export function useExecutionProgress(workflowId: string | null) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [totalSteps, setTotalSteps] = useState(0);

  const handleEvent = useCallback((event: ExecutionEvent) => {
    if (event.event === ExecutionEventType.PROGRESS) {
      const data = event.data as { progress: number; current_step: string; total_steps: number };
      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setTotalSteps(data.total_steps);
    }
  }, []);

  const stream = useExecutionStream(workflowId, {
    onEvent: handleEvent,
  });

  return {
    ...stream,
    progress,
    currentStep,
    totalSteps,
  };
}

export function useSubProblems(workflowId: string | null) {
  const [subProblems, setSubProblems] = useState<Array<{
    id: string;
    problem: string;
    status: string;
    solution?: string;
  }>>([]);

  const handleEvent = useCallback((event: ExecutionEvent) => {
    if (
      event.event === ExecutionEventType.SUBPROBLEM_CREATED ||
      event.event === ExecutionEventType.SUBPROBLEM_COMPLETED
    ) {
      const data = event.data as {
        subproblem_id: string;
        problem: string;
        status: string;
        solution?: string;
      };

      setSubProblems((prev) => {
        const existing = prev.find((p) => p.id === data.subproblem_id);
        if (existing) {
          return prev.map((p) =>
            p.id === data.subproblem_id
              ? { ...p, status: data.status, solution: data.solution }
              : p
          );
        } else {
          return [
            ...prev,
            {
              id: data.subproblem_id,
              problem: data.problem,
              status: data.status,
              solution: data.solution,
            },
          ];
        }
      });
    }
  }, []);

  const stream = useExecutionStream(workflowId, {
    onEvent: handleEvent,
  });

  return {
    ...stream,
    subProblems,
  };
}

export function useBubbles(workflowId: string | null) {
  const [bubbles, setBubbles] = useState<
    Record<
      string,
      {
        id: string;
        type: string;
        status: string;
        result?: any;
      }
    >
  >({});

  const handleEvent = useCallback((event: ExecutionEvent) => {
    if (
      event.event === ExecutionEventType.BUBBLE_ACTIVATED ||
      event.event === ExecutionEventType.BUBBLE_COMPLETED
    ) {
      const data = event.data as {
        bubble_id: string;
        bubble_type: string;
        status: string;
        result?: any;
      };

      setBubbles((prev) => ({
        ...prev,
        [data.bubble_id]: {
          id: data.bubble_id,
          type: data.bubble_type,
          status: data.status,
          result: data.result,
        },
      }));
    }
  }, []);

  const stream = useExecutionStream(workflowId, {
    onEvent: handleEvent,
  });

  return {
    ...stream,
    bubbles,
  };
}
