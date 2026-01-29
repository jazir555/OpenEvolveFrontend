/**
 * Real-time evolution updates hook
 */
export declare function useRealtimeEvolution(evolutionId?: string): {
    connectionState: import('../api').ConnectionState;
    data: any;
    error: Error;
    send: (type: import('../api').WebSocketMessageType, data: any) => void;
    on: (type: import('../api').WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Real-time adversarial testing updates hook
 */
export declare function useRealtimeAdversarial(testId?: string): {
    connectionState: import('../api').ConnectionState;
    data: any;
    error: Error;
    send: (type: import('../api').WebSocketMessageType, data: any) => void;
    on: (type: import('../api').WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Real-time monitoring updates hook
 */
export declare function useRealtimeMonitoring(enabled?: boolean): {
    connectionState: import('../api').ConnectionState;
    data: any;
    error: Error;
    send: (type: import('../api').WebSocketMessageType, data: any) => void;
    on: (type: import('../api').WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Auto-refresh hook for polling when WebSocket is not available
 */
export declare function useAutoRefresh(queryKey: string[], refreshFn: () => Promise<void>, interval?: number, enabled?: boolean): {
    startRefresh: () => void;
    stopRefresh: () => void;
    isRefreshing: boolean;
};
/**
 * Combined real-time hook that uses WebSocket when available,
 * falls back to polling
 */
export declare function useRealtime(type: 'evolution' | 'adversarial' | 'monitoring', id?: string, pollingFallback?: {
    queryKey: string[];
    refreshFn: () => Promise<void>;
    interval?: number;
}): {
    polling: {
        startRefresh: () => void;
        stopRefresh: () => void;
        isRefreshing: boolean;
    };
    refresh: () => Promise<void>;
    useWebSocket: boolean;
    connectionState: import('../api').ConnectionState;
    data: any;
    error: Error;
    send: (type: import('../api').WebSocketMessageType, data: any) => void;
    on: (type: import('../api').WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
