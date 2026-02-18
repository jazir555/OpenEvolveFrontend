import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { mockGetBubblelabsWorkflowInstance } = vi.hoisted(() => ({
  mockGetBubblelabsWorkflowInstance: vi.fn(),
}));

vi.mock('@/services/openevolveApi', () => ({
  openevolveApi: {
    getBubblelabsWorkflowInstance: mockGetBubblelabsWorkflowInstance,
  },
}));

import {
  isTerminalBubblelabsWorkflowStatus,
  useBubblelabsWorkflowPolling,
} from './useBubblelabsWorkflowPolling';

describe('useBubblelabsWorkflowPolling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockGetBubblelabsWorkflowInstance.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('polls workflow status until terminal state', async () => {
    mockGetBubblelabsWorkflowInstance
      .mockResolvedValueOnce({
        status: {
          instance_id: 'inst-1',
          status: 'running',
          current_stage: 'execute',
          progress: 20,
        },
        parameters: {},
      })
      .mockResolvedValueOnce({
        status: {
          instance_id: 'inst-1',
          status: 'completed',
          current_stage: 'done',
          progress: 100,
        },
        parameters: {},
      });

    const { result, unmount } = renderHook(() =>
      useBubblelabsWorkflowPolling({
        instanceId: 'inst-1',
        intervalMs: 100,
      })
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(mockGetBubblelabsWorkflowInstance).toHaveBeenCalledTimes(1);
    expect(result.current.status).toBe('running');

    await act(async () => {
      vi.advanceTimersByTime(120);
      await Promise.resolve();
    });

    expect(mockGetBubblelabsWorkflowInstance).toHaveBeenCalledTimes(2);
    expect(result.current.status).toBe('completed');

    act(() => {
      vi.advanceTimersByTime(400);
    });

    expect(mockGetBubblelabsWorkflowInstance).toHaveBeenCalledTimes(2);
    expect(result.current.isPolling).toBe(false);
    unmount();
  });

  it('does not poll when disabled', () => {
    renderHook(() =>
      useBubblelabsWorkflowPolling({
        instanceId: 'inst-1',
        enabled: false,
        intervalMs: 100,
      })
    );

    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(mockGetBubblelabsWorkflowInstance).not.toHaveBeenCalled();
  });

  it('supports manual refresh', async () => {
    mockGetBubblelabsWorkflowInstance.mockResolvedValue({
      status: {
        instance_id: 'inst-2',
        status: 'running',
        current_stage: 'execute',
        progress: 10,
      },
      parameters: {},
    });

    const { result } = renderHook(() =>
      useBubblelabsWorkflowPolling({
        instanceId: 'inst-2',
        enabled: false,
      })
    );

    await act(async () => {
      const response = await result.current.refresh();
      expect(response?.status.status).toBe('running');
    });

    expect(mockGetBubblelabsWorkflowInstance).toHaveBeenCalledTimes(1);
    expect(result.current.status).toBe('running');
  });
});

describe('isTerminalBubblelabsWorkflowStatus', () => {
  it('returns true for terminal statuses', () => {
    expect(isTerminalBubblelabsWorkflowStatus('completed')).toBe(true);
    expect(isTerminalBubblelabsWorkflowStatus('FAILED')).toBe(true);
    expect(isTerminalBubblelabsWorkflowStatus('cancelled')).toBe(true);
    expect(isTerminalBubblelabsWorkflowStatus('stopped')).toBe(true);
  });

  it('returns false for non-terminal statuses', () => {
    expect(isTerminalBubblelabsWorkflowStatus('created')).toBe(false);
    expect(isTerminalBubblelabsWorkflowStatus('running')).toBe(false);
    expect(isTerminalBubblelabsWorkflowStatus(undefined)).toBe(false);
  });
});
