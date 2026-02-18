import { useCallback, useEffect, useRef, useState } from 'react';
import {
  openevolveApi,
  type BubbleLabsWorkflowInstanceDetail,
} from '@/services/openevolveApi';

const TERMINAL_WORKFLOW_STATUSES = new Set(['completed', 'failed', 'cancelled', 'stopped']);

export const isTerminalBubblelabsWorkflowStatus = (status?: string): boolean =>
  !!status && TERMINAL_WORKFLOW_STATUSES.has(status.toLowerCase());

interface UseBubblelabsWorkflowPollingOptions {
  instanceId: string | null;
  enabled?: boolean;
  intervalMs?: number;
  stopOnTerminal?: boolean;
}

interface UseBubblelabsWorkflowPollingResult {
  detail: BubbleLabsWorkflowInstanceDetail | null;
  status: string | null;
  isLoading: boolean;
  isPolling: boolean;
  errorMessage: string | null;
  refresh: () => Promise<BubbleLabsWorkflowInstanceDetail | null>;
}

export function useBubblelabsWorkflowPolling(
  options: UseBubblelabsWorkflowPollingOptions
): UseBubblelabsWorkflowPollingResult {
  const {
    instanceId,
    enabled = true,
    intervalMs = 2000,
    stopOnTerminal = true,
  } = options;

  const [detail, setDetail] = useState<BubbleLabsWorkflowInstanceDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const latestStatusRef = useRef<string | null>(null);

  const refresh = useCallback(async (): Promise<BubbleLabsWorkflowInstanceDetail | null> => {
    if (!instanceId) {
      return null;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await openevolveApi.getBubblelabsWorkflowInstance(instanceId);
      setDetail(response);
      latestStatusRef.current = response.status?.status ?? null;
      return response;
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Failed to fetch workflow instance status'
      );
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [instanceId]);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setInterval> | null = null;

    const runPoll = async () => {
      if (!active || !instanceId) {
        return;
      }

      const response = await refresh();
      if (!active || !response) {
        return;
      }

      if (stopOnTerminal && isTerminalBubblelabsWorkflowStatus(response.status?.status)) {
        if (timer) {
          clearInterval(timer);
          timer = null;
        }
        setIsPolling(false);
      }
    };

    if (!enabled || !instanceId) {
      setIsPolling(false);
      return () => {
        active = false;
      };
    }

    setDetail(null);
    latestStatusRef.current = null;
    void runPoll();
    setIsPolling(true);

    timer = setInterval(() => {
      if (
        stopOnTerminal &&
        isTerminalBubblelabsWorkflowStatus(latestStatusRef.current ?? undefined)
      ) {
        if (timer) {
          clearInterval(timer);
          timer = null;
        }
        setIsPolling(false);
        return;
      }

      void runPoll();
    }, intervalMs);

    return () => {
      active = false;
      if (timer) {
        clearInterval(timer);
      }
      setIsPolling(false);
    };
  }, [enabled, instanceId, intervalMs, refresh, stopOnTerminal]);

  return {
    detail,
    status: detail?.status?.status ?? null,
    isLoading,
    isPolling,
    errorMessage,
    refresh,
  };
}
