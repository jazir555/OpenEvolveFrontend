import { ProgressUpdate } from './common';
export interface RequestConfig {
    url: string;
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
    headers?: Record<string, string>;
    body?: any;
    params?: Record<string, any>;
    timeout?: number;
}
export interface Response<T = any> {
    status: number;
    headers: Record<string, string>;
    data: T;
    message?: string;
}
export interface WebSocketMessage {
    type: string;
    payload: any;
    timestamp?: string;
}
export interface WebSocketConfig {
    url: string;
    protocols?: string | string[];
    reconnectAttempts?: number;
    reconnectDelay?: number;
}
export interface APIClientConfig {
    baseURL: string;
    apiKey?: string;
    timeout?: number;
    headers?: Record<string, string>;
    debug?: boolean;
    wsURL?: string;
}
export interface StreamExecutionOptions {
    onProgress: (update: ProgressUpdate) => void;
    onComplete?: (result: any) => void;
    onError?: (error: Error) => void;
    topic: string;
}
//# sourceMappingURL=api.d.ts.map