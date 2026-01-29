import { WebSocketClient, ConnectionState, WebSocketMessageType } from '../api/websocket';
/**
 * WebSocket connection hook
 */
export declare function useWebSocket(connectFn: (handlers: any) => WebSocketClient, enabled?: boolean): {
    connectionState: ConnectionState;
    data: any;
    error: Error;
    send: (type: WebSocketMessageType, data: any) => void;
    on: (type: WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Evolution WebSocket hook
 */
export declare function useEvolutionWebSocket(evolutionId?: string): {
    connectionState: ConnectionState;
    data: any;
    error: Error;
    send: (type: WebSocketMessageType, data: any) => void;
    on: (type: WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Adversarial testing WebSocket hook
 */
export declare function useAdversarialWebSocket(testId?: string): {
    connectionState: ConnectionState;
    data: any;
    error: Error;
    send: (type: WebSocketMessageType, data: any) => void;
    on: (type: WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Collaboration WebSocket hook
 */
export declare function useCollaborationWebSocket(roomId?: string): {
    sendContentUpdate: (content: string) => void;
    sendCursorUpdate: (position: {
        line: number;
        column: number;
    }) => void;
    sendComment: (comment: string, lineStart?: number, lineEnd?: number) => void;
    connectionState: ConnectionState;
    data: any;
    error: Error;
    send: (type: WebSocketMessageType, data: any) => void;
    on: (type: WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
/**
 * Monitoring WebSocket hook
 */
export declare function useMonitoringWebSocket(enabled?: boolean): {
    connectionState: ConnectionState;
    data: any;
    error: Error;
    send: (type: WebSocketMessageType, data: any) => void;
    on: (type: WebSocketMessageType, handler: (data: any) => void) => () => void;
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
};
