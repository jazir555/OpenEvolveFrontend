/**
 * WebSocket message types
 */
export type WebSocketMessageType = 'progress_update' | 'generation_complete' | 'evolution_complete' | 'attack_generated' | 'patch_generated' | 'patch_approved' | 'test_complete' | 'user_joined' | 'user_left' | 'content_update' | 'cursor_update' | 'comment_added' | 'resource_update' | 'service_status' | 'log_entry' | 'error' | 'connected' | 'disconnected';
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
export declare class WebSocketClient {
    private ws;
    private url;
    private reconnectInterval;
    private maxReconnectAttempts;
    private heartbeatInterval;
    private protocols?;
    private reconnectAttempts;
    private reconnectTimeout;
    private heartbeatTimeout;
    private handlers;
    private connectionState;
    private isManualClose;
    constructor(config: WebSocketConfig, handlers: WebSocketHandlers);
    /**
     * Connect to WebSocket server
     */
    connect(): void;
    /**
     * Disconnect from WebSocket server
     */
    disconnect(): void;
    /**
     * Send message through WebSocket
     */
    send(type: WebSocketMessageType, data: any): void;
    /**
     * Handle WebSocket open event
     */
    private handleOpen;
    /**
     * Handle WebSocket message event
     */
    private handleMessage;
    /**
     * Handle WebSocket error event
     */
    private handleError;
    /**
     * Handle WebSocket close event
     */
    private handleClose;
    /**
     * Schedule reconnection attempt
     */
    private scheduleReconnect;
    /**
     * Clear reconnect timeout
     */
    private clearReconnectTimeout;
    /**
     * Start heartbeat
     */
    private startHeartbeat;
    /**
     * Clear heartbeat
     */
    private clearHeartbeat;
    /**
     * Set connection state and notify handlers
     */
    private setConnectionState;
    /**
     * Get current connection state
     */
    getConnectionState(): ConnectionState;
    /**
     * Check if connected
     */
    isConnected(): boolean;
}
/**
 * Create a WebSocket client for evolution updates
 */
export declare function createEvolutionWebSocket(evolutionId: string, handlers: WebSocketHandlers): WebSocketClient;
/**
 * Create a WebSocket client for adversarial testing updates
 */
export declare function createAdversarialWebSocket(testId: string, handlers: WebSocketHandlers): WebSocketClient;
/**
 * Create a WebSocket client for collaboration
 */
export declare function createCollaborationWebSocket(roomId: string, handlers: WebSocketHandlers): WebSocketClient;
/**
 * Create a WebSocket client for system monitoring
 */
export declare function createMonitoringWebSocket(handlers: WebSocketHandlers): WebSocketClient;
