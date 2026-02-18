import { default as React } from 'react';
export interface IntegrationConfig {
    leanAideApiEndpoint: string;
    crewaiApiEndpoint: string;
    bubbleLabsApiEndpoint: string;
    researchQuestApiEndpoint: string;
    defaultTimeout: number;
    leanAideTimeout: number;
    crewaiTimeout: number;
    bubbleLabsTimeout: number;
    researchQuestTimeout: number;
    useAuthentication: boolean;
    authType: 'api_key' | 'oauth2' | 'jwt' | 'basic';
    apiKey: string;
    oauth2ClientId: string;
    oauth2ClientSecret: string;
    oauth2Scope: string;
    jwtToken: string;
    basicUsername: string;
    basicPassword: string;
    enableWebSocket: boolean;
    websocketUrl: string;
    websocketReconnectInterval: number;
    websocketMaxReconnectAttempts: number;
    websocketHeartbeatInterval: number;
    enableRetry: boolean;
    maxRetries: number;
    retryDelay: number;
    retryBackoffMultiplier: number;
    retryOnStatusCodes: number[];
    enableRateLimiting: boolean;
    maxRequestsPerSecond: number;
    maxRequestsPerMinute: number;
    rateLimitBurstSize: number;
    enableCircuitBreaker: boolean;
    circuitBreakerThreshold: number;
    circuitBreakerTimeout: number;
    circuitBreakerHalfOpenAttempts: number;
    enableCaching: boolean;
    cacheTtl: number;
    cacheSize: number;
    cacheStrategy: 'lru' | 'fifo' | 'lfu';
    enableLogging: boolean;
    logLevel: 'debug' | 'info' | 'warn' | 'error' | 'none';
    logRequests: boolean;
    logResponses: boolean;
    logErrors: boolean;
    enableHealthChecks: boolean;
    healthCheckInterval: number;
    healthCheckTimeout: number;
    healthCheckUnhealthyThreshold: number;
    leanAide: {
        provider: 'openai' | 'anthropic' | 'mistral' | 'custom';
        model: string;
        temperature: number;
        maxTokens: number;
    };
    crewai: {
        parallelExecution: boolean;
        maxParallelTasks: number;
        delegationTimeout: number;
    };
    bubbleLabs: {
        uiEnabled: boolean;
        realTimeUpdates: boolean;
        visualizationQuality: 'low' | 'medium' | 'high';
    };
    researchQuest: {
        knowledgeGraphEnabled: boolean;
        semanticSearch: boolean;
        maxResults: number;
    };
}
interface IntegrationConfigPanelProps {
    config: IntegrationConfig;
    onConfigChange: (config: IntegrationConfig) => void;
}
export declare const IntegrationConfigPanel: React.FC<IntegrationConfigPanelProps>;
export default IntegrationConfigPanel;
