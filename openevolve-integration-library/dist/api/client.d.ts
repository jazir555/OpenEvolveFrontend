import { BackendClient } from './backend';
import { ClientConfig, ExecutionOptions, ProgressUpdate, BatchRequest, BatchResult, HealthStatus, ConnectionState, RequestMetrics, RetryConfig } from './types';
import { IntegrationError } from './errors';
import { LeanAideIntegration, EvolutionIntegration, KnowledgeIntegration, MakerIntegration, HephaestusIntegration, DecompositionIntegration, VerificationIntegration, AssemblyIntegration, SolutionIntegration } from '../integrations';
export interface IntegrationRegistry {
    leanaide: LeanAideIntegration;
    evolution: EvolutionIntegration;
    knowledge: KnowledgeIntegration;
    maker: MakerIntegration;
    hephaestus: HephaestusIntegration;
    decomposition: DecompositionIntegration;
    verification: VerificationIntegration;
    assembly: AssemblyIntegration;
    solution: SolutionIntegration;
}
export declare enum IntegrationName {
    LEANAIDE = "leanaide",
    EVOLUTION = "evolution",
    KNOWLEDGE = "knowledge",
    MAKER = "maker",
    HEPHAESTUS = "hephaestus",
    DECOMPOSITION = "decomposition",
    VERIFICATION = "verification",
    ASSEMBLY = "assembly",
    SOLUTION = "solution"
}
export declare class OpenEvolveClient {
    static readonly VERSION = "1.1.0";
    private backend;
    private config;
    private integrationAdapters;
    private connectionState;
    private executionMetrics;
    private progressCallbacks;
    private errorHandlers;
    private retryConfig;
    private circuitBreakerConfig;
    private middleware;
    private debug;
    private healthCheckTimer;
    constructor(config: ClientConfig);
    private loadIntegrations;
    private setupWebSocket;
    startHealthCheck(interval: number): void;
    stopHealthCheck(): void;
    addErrorHandler(handler: (error: IntegrationError) => void): void;
    removeErrorHandler(handler: (error: IntegrationError) => void): void;
    private handleWebSocketMessage;
    private handleProgressUpdate;
    private handleExecutionComplete;
    private handleExecutionSuccess;
    private handleExecutionError;
    execute<TIntegration extends string, TInputs, TResult>(integration: TIntegration, inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    private runMiddleware;
    executeStream<TInputs, TResult>(integration: string, inputs: TInputs, onProgress: (update: ProgressUpdate) => void, options?: ExecutionOptions): Promise<TResult>;
    executeBatch<TInputs, TResult>(requests: BatchRequest<TInputs>[]): Promise<BatchResult<TResult>[]>;
    healthCheck(): Promise<HealthStatus>;
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    getVersions(): Record<string, string>;
    isConnected(): boolean;
    getConnectionState(): ConnectionState;
    private validateInputs;
    private getIntegration;
    get integrations(): IntegrationRegistry;
    getMetrics(executionId: string): RequestMetrics | null;
    getAllMetrics(): Map<string, RequestMetrics>;
    getMetricsSummary(): {
        totalRequests: number;
        successRate: number;
        averageDuration: number;
        totalRetries: number;
    };
    clearMetrics(): void;
    updateRetryConfig(config: Partial<RetryConfig>): void;
    private log;
    private clearOldMetrics;
    getBackend(): BackendClient;
}
export declare function createOpenEvolveClient(baseUrl: string): OpenEvolveClient;
export * from './types';
export * from './errors';
export { BackendClient } from './backend';
//# sourceMappingURL=client.d.ts.map