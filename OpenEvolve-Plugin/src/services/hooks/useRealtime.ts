// @ts-nocheck
import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  useEvolutionWebSocket,
  useAdversarialWebSocket,
  useMonitoringWebSocket,
  ConnectionState,
} from './useWebSocket';

/**
 * Real-time evolution updates hook
 */
export function useRealtimeEvolution(evolutionId?: string) {
  const queryClient = useQueryClient();
  const ws = useEvolutionWebSocket(evolutionId);

  /**
   * Handle progress updates
   */
  useEffect(() => {
    const unsubscribe = ws.on('progress_update', (data) => {
      // Update evolution state in cache
      queryClient.setQueryData(['evolution', evolutionId], (old: any) => {
        if (!old) return data;
        return {
          ...old,
          ...data,
        };
      });
    });

    return unsubscribe;
  }, [ws, evolutionId, queryClient]);

  /**
   * Handle generation complete
   */
  useEffect(() => {
    const unsubscribe = ws.on('generation_complete', (data) => {
      queryClient.invalidateQueries({ queryKey: ['evolution', evolutionId] });
    });

    return unsubscribe;
  }, [ws, evolutionId, queryClient]);

  /**
   * Handle evolution complete
   */
  useEffect(() => {
    const unsubscribe = ws.on('evolution_complete', (data) => {
      queryClient.invalidateQueries({ queryKey: ['evolution', evolutionId] });
      queryClient.invalidateQueries({ queryKey: ['evolutions'] });
    });

    return unsubscribe;
  }, [ws, evolutionId, queryClient]);

  return ws;
}

/**
 * Real-time adversarial testing updates hook
 */
export function useRealtimeAdversarial(testId?: string) {
  const queryClient = useQueryClient();
  const ws = useAdversarialWebSocket(testId);

  /**
   * Handle attack generated
   */
  useEffect(() => {
    const unsubscribe = ws.on('attack_generated', (data) => {
      queryClient.setQueryData(['adversarial', testId], (old: any) => {
        if (!old) return null;
        return {
          ...old,
          red_team_results: [...(old.red_team_results || []), data],
        };
      });
    });

    return unsubscribe;
  }, [ws, testId, queryClient]);

  /**
   * Handle patch generated
   */
  useEffect(() => {
    const unsubscribe = ws.on('patch_generated', (data) => {
      queryClient.setQueryData(['adversarial', testId], (old: any) => {
        if (!old) return null;
        return {
          ...old,
          blue_team_results: [...(old.blue_team_results || []), data],
          patches_generated: (old.patches_generated || 0) + 1,
        };
      });
    });

    return unsubscribe;
  }, [ws, testId, queryClient]);

  /**
   * Handle patch approved
   */
  useEffect(() => {
    const unsubscribe = ws.on('patch_approved', (data) => {
      queryClient.invalidateQueries({ queryKey: ['adversarial', testId] });
    });

    return unsubscribe;
  }, [ws, testId, queryClient]);

  /**
   * Handle test complete
   */
  useEffect(() => {
    const unsubscribe = ws.on('test_complete', (data) => {
      queryClient.invalidateQueries({ queryKey: ['adversarial', testId] });
      queryClient.invalidateQueries({ queryKey: ['adversarial-tests'] });
    });

    return unsubscribe;
  }, [ws, testId, queryClient]);

  return ws;
}

/**
 * Real-time monitoring updates hook
 */
export function useRealtimeMonitoring(enabled: boolean = true) {
  const queryClient = useQueryClient();
  const ws = useMonitoringWebSocket(enabled);

  /**
   * Handle resource updates
   */
  useEffect(() => {
    const unsubscribe = ws.on('resource_update', (data) => {
      queryClient.setQueryData(['monitoring', 'health'], (old: any) => {
        if (!old) return { resource_usage: data, active_operations: {} };
        return {
          ...old,
          resource_usage: data,
        };
      });
    });

    return unsubscribe;
  }, [ws, queryClient]);

  /**
   * Handle service status updates
   */
  useEffect(() => {
    const unsubscribe = ws.on('service_status', (data) => {
      queryClient.setQueryData(['monitoring', 'health'], (old: any) => {
        if (!old) return { services: data };
        return {
          ...old,
          services: data,
        };
      });
    });

    return unsubscribe;
  }, [ws, queryClient]);

  /**
   * Handle log entries
   */
  useEffect(() => {
    const unsubscribe = ws.on('log_entry', (data) => {
      // This could push to a log store or display in a log viewer component
      console.log('Log entry:', data);
    });

    return unsubscribe;
  }, [ws]);

  return ws;
}

/**
 * Auto-refresh hook for polling when WebSocket is not available
 */
export function useAutoRefresh(
  queryKey: string[],
  refreshFn: () => Promise<void>,
  interval: number = 5000,
  enabled: boolean = true
) {
  const queryClient = useQueryClient();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startRefresh = useCallback(() => {
    if (!enabled || intervalRef.current) return;

    intervalRef.current = setInterval(async () => {
      try {
        await refreshFn();
        queryClient.invalidateQueries({ queryKey });
      } catch (error) {
        console.error('Auto-refresh error:', error);
      }
    }, interval);
  }, [enabled, interval, refreshFn, queryKey, queryClient]);

  const stopRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    startRefresh();
    return stopRefresh;
  }, [startRefresh, stopRefresh]);

  return {
    startRefresh,
    stopRefresh,
    isRefreshing: !!intervalRef.current,
  };
}

/**
 * Combined real-time hook that uses WebSocket when available,
 * falls back to polling
 */
export function useRealtime(
  type: 'evolution' | 'adversarial' | 'monitoring',
  id?: string,
  pollingFallback?: {
    queryKey: string[];
    refreshFn: () => Promise<void>;
    interval?: number;
  }
) {
  const queryClient = useQueryClient();

  // WebSocket connection
  const ws =
    type === 'evolution'
      ? useRealtimeEvolution(id)
      : type === 'adversarial'
      ? useRealtimeAdversarial(id)
      : useRealtimeMonitoring(true);

  // Fallback polling when WebSocket is not connected
  const polling = useAutoRefresh(
    pollingFallback?.queryKey || [],
    pollingFallback?.refreshFn || (() => Promise.resolve()),
    pollingFallback?.interval || 5000,
    !ws.isConnected && !!pollingFallback
  );

  /**
   * Manual refresh function
   */
  const refresh = useCallback(async () => {
    if (ws.isConnected) {
      // WebSocket will auto-update, but we can invalidate cache
      queryClient.invalidateQueries({ queryKey: [type, id] });
    } else {
      // Manual refresh when polling
      await pollingFallback?.refreshFn?.();
      queryClient.invalidateQueries({ queryKey: pollingFallback?.queryKey });
    }
  }, [ws.isConnected, type, id, polling, pollingFallback, queryClient]);

  return {
    ...ws,
    polling,
    refresh,
    useWebSocket: ws.isConnected,
  };
}
