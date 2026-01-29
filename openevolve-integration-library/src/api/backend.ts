/**
 * Backend Client for OpenEvolve Integration Library
 *
 * Handles all communication with the Python backend service
 */

import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import { io, Socket } from 'socket.io-client';
import {
  BackendStatus,
  WebSocketMessage,
  RequestTransform,
  ResponseTransform,
} from './types';
import {
  createIntegrationError,
} from './errors';


/**
 * Backend client configuration
 */
export interface BackendClientConfig {
  /** Base URL for the backend */
  baseUrl: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** API key for authentication */
  apiKey?: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Request interceptor */
  requestTransform?: RequestTransform;
  /** Response interceptor */
  responseTransform?: ResponseTransform;
}

/**
 * WebSocket event handlers
 */
export interface WebSocketHandlers {
  /** Connection established */
  onConnect?: () => void;
  /** Connection closed */
  onDisconnect?: (reason: string) => void;
  /** Connection error */
  onError?: (error: Error) => void;
  /** Message received */
  onMessage?: (message: WebSocketMessage) => void;
  /** Reconnection attempt */
  onReconnect?: (attemptNumber: number) => void;
}

/**
 * Client for communicating with OpenEvolve Python backend
 */
export class BackendClient {
  private httpClient: AxiosInstance;
  private wsUrl: string;
  private socket: Socket | null = null;
  private config: BackendClientConfig;
  private debug: boolean;
  private globalAbortController: AbortController = new AbortController();

  /**
   * Create a new backend client
   *
   * @param config - Client configuration
   */
  constructor(config: BackendClientConfig) {
    this.config = config;
    this.debug = config.debug || false;

    // Initialize HTTP client
    this.httpClient = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(config.apiKey && { Authorization: `Bearer ${config.apiKey}` }),
        ...config.headers,
      },
    });

    // Use global abort controller by default
    this.httpClient.defaults.signal = this.globalAbortController.signal;

    // Setup interceptors
    this.setupInterceptors();

    // Derive WebSocket URL from base URL, ensuring no trailing slash
    const sanitizedBaseUrl = config.baseUrl.endsWith('/') 
      ? config.baseUrl.slice(0, -1) 
      : config.baseUrl;

    this.wsUrl = sanitizedBaseUrl
      .replace('http://', 'ws://')
      .replace('https://', 'wss://');

    this.log('Backend client initialized', { baseUrl: sanitizedBaseUrl });
  }

  /**
   * Setup request and response interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor
    this.httpClient.interceptors.request.use(
      (config) => {
        this.log('Request:', {
          method: config.method?.toUpperCase(),
          url: config.url,
          data: config.data,
        });

        // Apply custom transform if provided
        if (this.config.requestTransform && config.data) {
          config.data = this.config.requestTransform(config.data);
        }

        return config;
      },
      (error) => {
        this.log('Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.httpClient.interceptors.response.use(
      (response) => {
        this.log('Response:', {
          status: response.status,
          data: response.data,
        });

        // Apply custom transform if provided
        if (this.config.responseTransform && response.data) {
          response.data = this.config.responseTransform(response.data);
        }

        return response;
      },
      (error) => {
        this.log('Response error:', error);
        return Promise.reject(this.handleAxiosError(error));
      }
    );
  }

  /**
   * Handle axios errors
   */
  private handleAxiosError(error: any): Error {
    return createIntegrationError('backend', error);
  }



  /**
   * Make a POST request
   *
   * @param endpoint - API endpoint
   * @param data - Request data
   * @param config - Axios config override
   * @returns Response data
   */
  async post<TRequest, TResponse>(
    endpoint: string,
    data: TRequest,
    config?: AxiosRequestConfig
  ): Promise<TResponse> {
    try {
      const response = await this.httpClient.post<TResponse>(
        endpoint,
        data,
        config
      );
      return response.data;
    } catch (error) {
      throw this.handleAxiosError(error);
    }
  }




  /**
   * Make a GET request
   *
   * @param endpoint - API endpoint
   * @param config - Axios config override
   * @returns Response data
   */
  async get<TResponse>(
    endpoint: string,
    config?: AxiosRequestConfig
  ): Promise<TResponse> {
    try {
      const response = await this.httpClient.get<TResponse>(endpoint, config);
      return response.data;
    } catch (error) {
      throw this.handleAxiosError(error);
    }
  }

  /**
   * Make a PUT request
   *
   * @param endpoint - API endpoint
   * @param data - Request data
   * @param config - Axios config override
   * @returns Response data
   */
  async put<TRequest, TResponse>(
    endpoint: string,
    data: TRequest,
    config?: AxiosRequestConfig
  ): Promise<TResponse> {
    try {
      const response = await this.httpClient.put<TResponse>(
        endpoint,
        data,
        config
      );
      return response.data;
    } catch (error) {
      throw this.handleAxiosError(error);
    }
  }

  /**
   * Make a DELETE request
   *
   * @param endpoint - API endpoint
   * @param config - Axios config override
   * @returns Response data
   */
  async delete<TResponse>(
    endpoint: string,
    config?: AxiosRequestConfig
  ): Promise<TResponse> {
    try {
      const response = await this.httpClient.delete<TResponse>(
        endpoint,
        config
      );
      return response.data;
    } catch (error) {
      throw this.handleAxiosError(error);
    }
  }

  /**
   * Make a PATCH request
   *
   * @param endpoint - API endpoint
   * @param data - Request data
   * @param config - Axios config override
   * @returns Response data
   */
  async patch<TRequest, TResponse>(
    endpoint: string,
    data: TRequest,
    config?: AxiosRequestConfig
  ): Promise<TResponse> {
    try {
      const response = await this.httpClient.patch<TResponse>(
        endpoint,
        data,
        config
      );
      return response.data;
    } catch (error) {
      throw this.handleAxiosError(error);
    }
  }



  /**
   * Establish WebSocket connection
   *
   * @param path - WebSocket path (default: /ws)
   * @param handlers - Event handlers
   * @returns WebSocket socket
   */
  websocket(
    path: string = '/ws',
    handlers?: WebSocketHandlers
  ): Socket {
    const isDefaultPath = path === '/ws';

    try {
      if (isDefaultPath && this.socket?.connected) {
        this.log('WebSocket already connected');
        // If we have new handlers, we should clear old ones and attach new ones
        if (handlers) {
          this.socket.off('connect');
          this.socket.off('disconnect');
          this.socket.off('error');
          this.socket.off('message');
          this.socket.off('reconnect');
          this.attachHandlers(this.socket, handlers);
        }
        return this.socket;
      }

      this.log('Connecting WebSocket:', { url: this.wsUrl + path });

      const socket = io(this.wsUrl + path, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        timeout: 10000,
        auth: this.config.apiKey ? { apiKey: this.config.apiKey } : undefined,
      });

      if (isDefaultPath) {
        this.socket = socket;
      }

      // Setup event handlers
      if (handlers) {
        this.attachHandlers(socket, handlers);
      }

      return socket;
    } catch (error) {
      this.log('Failed to initialize WebSocket:', error);
      // Return a "fake" socket-like object that won't crash when called but is effectively dead
      // or just rethrow as a ConnectionError
      throw createIntegrationError('backend', {
        message: 'Failed to initialize WebSocket connection',
        originalError: error
      });
    }
  }


  /**
   * Attach handlers to socket
   */
  private attachHandlers(socket: Socket, handlers: WebSocketHandlers): void {
    socket.on('connect', () => {
      this.log('WebSocket connected');
      handlers.onConnect?.();
    });

    socket.on('disconnect', (reason) => {
      this.log('WebSocket disconnected:', { reason });
      handlers.onDisconnect?.(reason);
    });

    socket.on('error', (error) => {
      this.log('WebSocket error:', error);
      handlers.onError?.(error);
    });

    socket.on('message', (message: WebSocketMessage) => {
      this.log('WebSocket message:', message);
      handlers.onMessage?.(message);
    });

    socket.on('reconnect', (attemptNumber) => {
      this.log('WebSocket reconnected:', { attemptNumber });
      handlers.onReconnect?.(attemptNumber);
    });
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.socket) {
      this.log('Disconnecting WebSocket');
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isWebSocketConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Ping the backend
   *
   * @returns True if backend is responsive
   */
  async ping(): Promise<boolean> {
    try {
      const start = Date.now();
      await this.get('/ping');
      const duration = Date.now() - start;
      this.log('Ping successful:', { duration: `${duration}ms` });
      return true;
    } catch (error) {
      this.log('Ping failed:', error);
      return false;
    }
  }

  /**
   * Get backend status
   *
   * @returns Backend status information
   */
  async getStatus(): Promise<BackendStatus> {
    try {
      const status = await this.get<BackendStatus>('/status');
      this.log('Backend status:', status);
      return status;
    } catch (error) {
      throw this.handleAxiosError(error as AxiosError);
    }
  }

  /**
   * Get backend health
   *
   * @returns Health check results
   */
  async getHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    checks: Record<string, boolean>;
  }> {
    try {
      const health = await this.get<{
        status: 'healthy' | 'degraded' | 'unhealthy';
        checks: Record<string, boolean>;
      }>('/health');
      this.log('Backend health:', health);
      return health;
    } catch (error) {
      throw this.handleAxiosError(error as AxiosError);
    }
  }

  /**
   * Get backend version
   *
   * @returns Version string
   */
  async getVersion(): Promise<string> {
    try {
      const { version } = await this.get<{ version: string }>('/version');
      this.log('Backend version:', version);
      return version;
    } catch (error) {
      throw this.handleAxiosError(error as AxiosError);
    }
  }

  /**
   * Cancel all pending requests
   */
  cancelAllRequests(): void {
    this.globalAbortController.abort();
    // Create a new one for future requests
    this.globalAbortController = new AbortController();
    this.httpClient.defaults.signal = this.globalAbortController.signal;
    this.log('All requests cancelled');
  }

  /**
   * Update client configuration
   *
   * @param config - New configuration
   */
  updateConfig(config: Partial<BackendClientConfig>): void {
    this.config = { ...this.config, ...config };

    // Update HTTP client defaults
    if (config.timeout) {
      this.httpClient.defaults.timeout = config.timeout;
    }

    if (config.headers) {
      this.httpClient.defaults.headers = {
        ...this.httpClient.defaults.headers,
        ...config.headers,
      };
    }

    if (config.apiKey) {
      this.httpClient.defaults.headers['Authorization'] = `Bearer ${config.apiKey}`;
    }

    this.log('Configuration updated:', config);
  }

  /**
   * Log debug message
   */
  public log(message: string, data?: any): void {
    if (this.debug) {
      if (data) {
        console.log(`[BackendClient] ${message}`, data);
      } else {
        console.log(`[BackendClient] ${message}`);
      }
    }
  }

  /**
   * Get the underlying axios instance
   *
   * @returns Axios instance
   */
  getHttpClient(): AxiosInstance {
    return this.httpClient;
  }

  /**
   * Get the WebSocket socket
   *
   * @returns Socket instance or null
   */
  getSocket(): Socket | null {
    return this.socket;
  }
}

/**
 * Create a backend client with default configuration
 *
 * @param baseUrl - Backend base URL
 * @returns Backend client instance
 */
export function createBackendClient(
  baseUrl: string
): BackendClient {
  return new BackendClient({
    baseUrl,
    timeout: 30000,
    debug: false,
  });
}
