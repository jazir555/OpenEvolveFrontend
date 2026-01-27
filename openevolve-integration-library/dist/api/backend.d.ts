import { AxiosInstance, AxiosRequestConfig } from 'axios';
import { Socket } from 'socket.io-client';
import { BackendStatus, WebSocketMessage, RequestTransform, ResponseTransform } from './types';
export interface BackendClientConfig {
    baseUrl: string;
    timeout?: number;
    apiKey?: string;
    debug?: boolean;
    headers?: Record<string, string>;
    requestTransform?: RequestTransform;
    responseTransform?: ResponseTransform;
}
export interface WebSocketHandlers {
    onConnect?: () => void;
    onDisconnect?: (reason: string) => void;
    onError?: (error: Error) => void;
    onMessage?: (message: WebSocketMessage) => void;
    onReconnect?: (attemptNumber: number) => void;
}
export declare class BackendClient {
    private httpClient;
    private wsUrl;
    private socket;
    private config;
    private debug;
    private globalAbortController;
    constructor(config: BackendClientConfig);
    private setupInterceptors;
    private handleAxiosError;
    post<TRequest, TResponse>(endpoint: string, data: TRequest, config?: AxiosRequestConfig): Promise<TResponse>;
    get<TResponse>(endpoint: string, config?: AxiosRequestConfig): Promise<TResponse>;
    put<TRequest, TResponse>(endpoint: string, data: TRequest, config?: AxiosRequestConfig): Promise<TResponse>;
    delete<TResponse>(endpoint: string, config?: AxiosRequestConfig): Promise<TResponse>;
    patch<TRequest, TResponse>(endpoint: string, data: TRequest, config?: AxiosRequestConfig): Promise<TResponse>;
    websocket(path?: string, handlers?: WebSocketHandlers): Socket;
    private attachHandlers;
    disconnectWebSocket(): void;
    isWebSocketConnected(): boolean;
    ping(): Promise<boolean>;
    getStatus(): Promise<BackendStatus>;
    getHealth(): Promise<{
        status: 'healthy' | 'degraded' | 'unhealthy';
        checks: Record<string, boolean>;
    }>;
    getVersion(): Promise<string>;
    cancelAllRequests(): void;
    updateConfig(config: Partial<BackendClientConfig>): void;
    log(message: string, data?: any): void;
    getHttpClient(): AxiosInstance;
    getSocket(): Socket | null;
}
export declare function createBackendClient(baseUrl: string): BackendClient;
//# sourceMappingURL=backend.d.ts.map