// @ts-nocheck
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  createEvolutionWebSocket,
  createAdversarialWebSocket,
  createCollaborationWebSocket,
  createMonitoringWebSocket,
  WebSocketClient,
  WebSocketMessage,
  ConnectionState,
  WebSocketMessageType,
} from '@/services/api/websocket';

type WebSocketUrlOptions = {
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Error) => void;
  reconnectInterval?: number;
  reconnectAttempts?: number;
};

/**
 * WebSocket connection hook
 */
export function useWebSocket(
  connectFnOrUrl: ((handlers: any) => WebSocketClient) | string,
  enabledOrOptions: boolean | WebSocketUrlOptions = true
) {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<Error | null>(null);
  const wsClientRef = useRef<WebSocketClient | null>(null);
  const messageHandlersRef = useRef<Map<WebSocketMessageType, Array<(data: any) => void>>>(new Map());

  const isUrlMode = typeof connectFnOrUrl === 'string';
  const enabled = typeof enabledOrOptions === 'boolean' ? enabledOrOptions : true;
  const urlOptions = typeof enabledOrOptions === 'object' ? enabledOrOptions : undefined;

  const connectFn = useCallback(
    (handlers: any) => {
      if (typeof connectFnOrUrl === 'function') {
        return connectFnOrUrl(handlers);
      }
      return new WebSocketClient(
        {
          url: connectFnOrUrl,
          reconnectInterval: urlOptions?.reconnectInterval,
          maxReconnectAttempts: urlOptions?.reconnectAttempts,
        },
        {
          ...handlers,
          onMessage: (message: WebSocketMessage) => {
            handlers.onMessage?.(message);
            urlOptions?.onMessage?.(message);
          },
          onError: (err: Error) => {
            handlers.onError?.(err);
            urlOptions?.onError?.(err);
          },
        }
      );
    },
    [connectFnOrUrl, urlOptions?.onMessage, urlOptions?.onError, urlOptions?.reconnectAttempts, urlOptions?.reconnectInterval]
  );

  /**
   * Register a message handler
   */
  const on = useCallback((type: WebSocketMessageType, handler: (data: any) => void) => {
    const handlers = messageHandlersRef.current.get(type) || [];
    messageHandlersRef.current.set(type, [...handlers, handler]);

    // Return cleanup function
    return () => {
      const currentHandlers = messageHandlersRef.current.get(type) || [];
      messageHandlersRef.current.set(
        type,
        currentHandlers.filter((h) => h !== handler)
      );
    };
  }, []);

  /**
   * Send message through WebSocket
   */
  const send = useCallback((typeOrMessage: WebSocketMessageType | WebSocketMessage, data?: any) => {
    if (typeof typeOrMessage === 'string') {
      wsClientRef.current?.send(typeOrMessage, data);
      return;
    }
    wsClientRef.current?.send(typeOrMessage.type, typeOrMessage.data);
  }, []);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    if (!enabled) return;

    const handlers = {
      onMessage: (message: WebSocketMessage) => {
        setData(message);

        // Call registered message handlers
        const handlers = messageHandlersRef.current.get(message.type) || [];
        handlers.forEach((handler) => handler(message.data));
      },
      onConnectionChange: (state: ConnectionState) => {
        setConnectionState(state);
      },
      onError: (err: Error) => {
        setError(err);
      },
    };

    wsClientRef.current = connectFn(handlers);
    wsClientRef.current.connect();
  }, [enabled, connectFn]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    wsClientRef.current?.disconnect();
    wsClientRef.current = null;
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    connectionState,
    data,
    error,
    send,
    on,
    isConnected: connectionState === 'connected',
    connect,
    disconnect,
    ws: isUrlMode ? wsClientRef.current?.getSocket?.() : undefined,
  };
}

/**
 * Evolution WebSocket hook
 */
export function useEvolutionWebSocket(evolutionId?: string) {
  return useWebSocket(
    useCallback(
      (handlers) => {
        if (!evolutionId) {
          throw new Error('Evolution ID is required');
        }
        return createEvolutionWebSocket(evolutionId, handlers);
      },
      [evolutionId]
    ),
    !!evolutionId
  );
}

/**
 * Adversarial testing WebSocket hook
 */
export function useAdversarialWebSocket(testId?: string) {
  return useWebSocket(
    useCallback(
      (handlers) => {
        if (!testId) {
          throw new Error('Test ID is required');
        }
        return createAdversarialWebSocket(testId, handlers);
      },
      [testId]
    ),
    !!testId
  );
}

/**
 * Collaboration WebSocket hook
 */
export function useCollaborationWebSocket(roomId?: string) {
  const ws = useWebSocket(
    useCallback(
      (handlers) => {
        if (!roomId) {
          throw new Error('Room ID is required');
        }
        return createCollaborationWebSocket(roomId, handlers);
      },
      [roomId]
    ),
    !!roomId
  );

  /**
   * Send content update
   */
  const sendContentUpdate = useCallback((content: string) => {
    ws.send('content_update', { content });
  }, [ws]);

  /**
   * Send cursor update
   */
  const sendCursorUpdate = useCallback((position: { line: number; column: number }) => {
    ws.send('cursor_update', { position });
  }, [ws]);

  /**
   * Send comment
   */
  const sendComment = useCallback((comment: string, lineStart?: number, lineEnd?: number) => {
    ws.send('comment_added', { comment, line_start: lineStart, line_end: lineEnd });
  }, [ws]);

  return {
    ...ws,
    sendContentUpdate,
    sendCursorUpdate,
    sendComment,
  };
}

/**
 * Monitoring WebSocket hook
 */
export function useMonitoringWebSocket(enabled: boolean = true) {
  return useWebSocket(
    useCallback((handlers) => createMonitoringWebSocket(handlers), []),
    enabled
  );
}
