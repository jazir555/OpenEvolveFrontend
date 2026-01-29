import { errorLogger } from '@/utils';

/**
 * WebSocket message types
 */
export type WebSocketMessageType =
  | 'progress_update'
  | 'generation_complete'
  | 'evolution_complete'
  | 'attack_generated'
  | 'patch_generated'
  | 'patch_approved'
  | 'test_complete'
  | 'user_joined'
  | 'user_left'
  | 'content_update'
  | 'cursor_update'
  | 'comment_added'
  | 'resource_update'
  | 'service_status'
  | 'log_entry'
  | 'error'
  | 'connected'
  | 'disconnected';

/**
 * WebSocket message interface
 */
export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  data: T;
  timestamp?: string;
}

/**
 * WebSocket connection state
 */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

/**
 * WebSocket client configuration
 */
export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  protocols?: string | string[];
}

/**
 * WebSocket event handlers
 */
export interface WebSocketHandlers {
  onMessage?: (message: WebSocketMessage) => void;
  onConnectionChange?: (state: ConnectionState) => void;
  onError?: (error: Error) => void;
}

/**
 * WebSocket client class
 */
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number;
  private maxReconnectAttempts: number;
  private heartbeatInterval: number;
  private protocols?: string | string[];
  private reconnectAttempts: number = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatTimeout: NodeJS.Timeout | null = null;
  private handlers: WebSocketHandlers;
  private connectionState: ConnectionState = 'disconnected';
  private isManualClose: boolean = false;

  constructor(config: WebSocketConfig, handlers: WebSocketHandlers) {
    this.url = config.url;
    this.reconnectInterval = config.reconnectInterval || 3000;
    this.maxReconnectAttempts = config.maxReconnectAttempts || 10;
    this.heartbeatInterval = config.heartbeatInterval || 30000;
    this.protocols = config.protocols;
    this.handlers = handlers;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    try {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        return;
      }

      this.isManualClose = false;
      this.setConnectionState('connecting');

      this.ws = new WebSocket(this.url, this.protocols);

      this.ws.onopen = this.handleOpen;
      this.ws.onmessage = this.handleMessage;
      this.ws.onerror = this.handleError;
      this.ws.onclose = this.handleClose;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'connect',
        additionalData: { url: this.url }
      });
      this.handlers.onError?.(err);
      this.setConnectionState('error');
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    try {
      this.isManualClose = true;
      this.clearReconnectTimeout();
      this.clearHeartbeat();

      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }

      this.setConnectionState('disconnected');
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'disconnect',
        additionalData: { url: this.url }
      });
    }
  }

  /**
   * Send message through WebSocket
   */
  send(type: WebSocketMessageType, data: any): void {
    try {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const message: WebSocketMessage = {
          type,
          data,
          timestamp: new Date().toISOString(),
        };
        this.ws.send(JSON.stringify(message));
      } else {
        console.warn('WebSocket is not connected. Cannot send message.');
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'send',
        additionalData: { type, url: this.url }
      });
    }
  }

  /**
   * Access underlying WebSocket instance (primarily for testing).
   */
  getSocket(): WebSocket | null {
    return this.ws;
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen = () => {
    try {
      this.setConnectionState('connected');
      this.reconnectAttempts = 0;
      this.clearReconnectTimeout();
      this.startHeartbeat();

      // Send connected message
      this.handlers.onMessage?.({
        type: 'connected',
        data: { url: this.url },
      });
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'handleOpen',
        additionalData: { url: this.url }
      });
    }
  };

  /**
   * Handle WebSocket message event
   */
  private handleMessage = (event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.handlers.onMessage?.(message);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'handleMessage',
        additionalData: { url: this.url }
      });
      console.error('Failed to parse WebSocket message:', error);
    }
  };

  /**
   * Handle WebSocket error event
   */
  private handleError = (error: Event) => {
    try {
      console.error('WebSocket error:', error);
      const err = new Error('WebSocket connection error');
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'handleError',
        additionalData: { url: this.url }
      });
      this.handlers.onError?.(err);
      this.setConnectionState('error');
    } catch (handlerError) {
      const err = handlerError instanceof Error ? handlerError : new Error(String(handlerError));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'handleError',
        additionalData: { url: this.url }
      });
    }
  };

  /**
   * Handle WebSocket close event
   */
  private handleClose = () => {
    try {
      this.setConnectionState('disconnected');
      this.clearHeartbeat();

      // Notify about disconnection
      this.handlers.onMessage?.({
        type: 'disconnected',
        data: { url: this.url },
      });

      // Attempt to reconnect if not manually closed
      if (!this.isManualClose) {
        this.scheduleReconnect();
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'handleClose',
        additionalData: { url: this.url }
      });
    }
  };

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    try {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnect attempts reached');
        errorLogger.logError(
          new Error('Max reconnect attempts reached'),
          'error',
          {
            component: 'WebSocketClient',
            function: 'scheduleReconnect',
            additionalData: {
              url: this.url,
              attempts: this.reconnectAttempts,
              maxAttempts: this.maxReconnectAttempts
            }
          }
        );
        return;
      }

      this.clearReconnectTimeout();
      this.reconnectTimeout = setTimeout(() => {
        try {
          this.reconnectAttempts++;
          console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
          this.connect();
        } catch (error) {
          const err = error instanceof Error ? error : new Error(String(error));
          errorLogger.logError(err, 'error', {
            component: 'WebSocketClient',
            function: 'reconnectTimeoutCallback',
            additionalData: {
              url: this.url,
              attempt: this.reconnectAttempts
            }
          });
        }
      }, this.reconnectInterval);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'scheduleReconnect',
        additionalData: { url: this.url }
      });
    }
  }

  /**
   * Clear reconnect timeout
   */
  private clearReconnectTimeout(): void {
    try {
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'clearReconnectTimeout',
        additionalData: { url: this.url }
      });
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    try {
      this.clearHeartbeat();
      this.heartbeatTimeout = setInterval(() => {
        try {
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'ping' }));
          }
        } catch (error) {
          const err = error instanceof Error ? error : new Error(String(error));
          errorLogger.logError(err, 'error', {
            component: 'WebSocketClient',
            function: 'heartbeatSend',
            additionalData: { url: this.url }
          });
        }
      }, this.heartbeatInterval);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'startHeartbeat',
        additionalData: { url: this.url }
      });
    }
  }

  /**
   * Clear heartbeat
   */
  private clearHeartbeat(): void {
    try {
      if (this.heartbeatTimeout) {
        clearInterval(this.heartbeatTimeout);
        this.heartbeatTimeout = null;
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'clearHeartbeat',
        additionalData: { url: this.url }
      });
    }
  }

  /**
   * Set connection state and notify handlers
   */
  private setConnectionState(state: ConnectionState): void {
    try {
      this.connectionState = state;
      this.handlers.onConnectionChange?.(state);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(err, 'error', {
        component: 'WebSocketClient',
        function: 'setConnectionState',
        additionalData: { url: this.url, state }
      });
    }
  }

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected' &&
           this.ws !== null &&
           this.ws.readyState === WebSocket.OPEN;
  }
}

/**
 * Create a WebSocket client for evolution updates
 */
export function createEvolutionWebSocket(
  evolutionId: string,
  handlers: WebSocketHandlers
): WebSocketClient {
  try {
    const wsUrl = `${getWebSocketBaseURL()}/ws/evolution/${evolutionId}`;
    return new WebSocketClient({ url: wsUrl }, handlers);
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(err, 'error', {
      component: 'WebSocketClient',
      function: 'createEvolutionWebSocket',
      additionalData: { evolutionId }
    });
    throw err;
  }
}

/**
 * Create a WebSocket client for adversarial testing updates
 */
export function createAdversarialWebSocket(
  testId: string,
  handlers: WebSocketHandlers
): WebSocketClient {
  try {
    const wsUrl = `${getWebSocketBaseURL()}/ws/adversarial/${testId}`;
    return new WebSocketClient({ url: wsUrl }, handlers);
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(err, 'error', {
      component: 'WebSocketClient',
      function: 'createAdversarialWebSocket',
      additionalData: { testId }
    });
    throw err;
  }
}

/**
 * Create a WebSocket client for collaboration
 */
export function createCollaborationWebSocket(
  roomId: string,
  handlers: WebSocketHandlers
): WebSocketClient {
  try {
    const wsUrl = `${getWebSocketBaseURL()}/ws/collaboration/${roomId}`;
    return new WebSocketClient({ url: wsUrl }, handlers);
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(err, 'error', {
      component: 'WebSocketClient',
      function: 'createCollaborationWebSocket',
      additionalData: { roomId }
    });
    throw err;
  }
}

/**
 * Create a WebSocket client for system monitoring
 */
export function createMonitoringWebSocket(
  handlers: WebSocketHandlers
): WebSocketClient {
  try {
    const wsUrl = `${getWebSocketBaseURL()}/ws/monitoring`;
    return new WebSocketClient({ url: wsUrl }, handlers);
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(err, 'error', {
      component: 'WebSocketClient',
      function: 'createMonitoringWebSocket'
    });
    throw err;
  }
}

/**
 * Get WebSocket base URL from environment
 */
function getWebSocketBaseURL(): string {
  try {
    const httpUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
    const normalized = /^https?:\/\//i.test(httpUrl)
      ? httpUrl
      : (typeof window !== 'undefined' && window.location?.origin
          ? new URL(httpUrl, window.location.origin).toString()
          : httpUrl);
    return normalized.replace(/^http/, 'ws');
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(err, 'error', {
      component: 'WebSocketClient',
      function: 'getWebSocketBaseURL'
    });
    return 'ws://localhost:8000'; // fallback URL
  }
}
