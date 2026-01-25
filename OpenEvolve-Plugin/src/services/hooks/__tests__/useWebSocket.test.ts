/**
 * Tests for useWebSocket hook
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useWebSocket } from '../useWebSocket';
import { MockWebSocket } from '@/test/utils/test-utils';

// Mock global WebSocket
global.WebSocket = MockWebSocket as any;

describe('useWebSocket', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Cleanup any open connections
    vi.restoreAllMocks();
  });

  it('should connect to WebSocket', async () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws/test')
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });

  it('should send messages', async () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws/test')
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    act(() => {
      result.current.send({ type: 'test', data: 'hello' });
    });

    // Message should be sent
    const ws = (result.current as any).ws;
    expect(ws?.getSentMessages()).toContainEqual({
      type: 'test',
      data: 'hello',
    });
  });

  it('should receive messages', async () => {
    const onMessage = vi.fn();

    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws/test', { onMessage })
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    act(() => {
      const ws = (result.current as any).ws;
      ws.receiveMessage({ type: 'update', data: { progress: 50 } });
    });

    expect(onMessage).toHaveBeenCalledWith({
      type: 'update',
      data: { progress: 50 },
    });
  });

  it('should handle connection errors', async () => {
    const onError = vi.fn();

    renderHook(() =>
      useWebSocket('ws://invalid-host:8000/ws/test', { onError })
    );

    await waitFor(
      () => {
        // Error should be handled
        expect(onError).toHaveBeenCalled();
      },
      { timeout: 5000 }
    );
  });

  it('should reconnect on disconnect', async () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws/test', {
        reconnectInterval: 100,
        reconnectAttempts: 3,
      })
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    // Simulate disconnect
    act(() => {
      const ws = (result.current as any).ws;
      ws.close();
    });

    // Should attempt to reconnect
    await waitFor(
      () => {
        expect(result.current.isConnected).toBe(true);
      },
      { timeout: 2000 }
    );
  });

  it('should stop reconnecting after max attempts', async () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://invalid-host:8000/ws/test', {
        reconnectInterval: 100,
        reconnectAttempts: 2,
      })
    );

    // Should fail to connect
    await waitFor(
      () => {
        expect(result.current.isConnected).toBe(false);
      },
      { timeout: 5000 }
    );
  });

  it('should cleanup on unmount', async () => {
    const { result, unmount } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws/test')
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    const ws = (result.current as any).ws;
    const closeSpy = vi.spyOn(ws, 'close');

    unmount();

    expect(closeSpy).toHaveBeenCalled();
  });
});
