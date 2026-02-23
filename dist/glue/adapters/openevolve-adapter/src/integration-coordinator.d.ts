/**
 * Integration Coordinator
 *
 * Coordinates all adapters in the OpenEvolve federation:
 * - Z3 Prover Adapter (formal verification)
 * - LeanAide Adapter (proof assistant)
 * - RAGBits Adapter (retrieval augmented generation)
 * - Vector DB Adapter (vector storage)
 * - Graphiti Adapter (graph knowledge)
 * - KarateClub Adapter (graph ML)
 *
 * The coordinator provides:
 * - Unified interface for all integrations
 * - Intelligent routing based on problem type
 * - Load balancing across adapters
 * - Health monitoring and failover
 * - Canonical schema transformation (Anti-Corruption Layer)
 *
 * Environment Variables:
 *   Z3_ADAPTER_URL - Z3 adapter endpoint
 *   LEANAIDE_ADAPTER_URL - LeanAide adapter endpoint
 *   RAGBITS_ADAPTER_URL - RAGBits adapter endpoint
 *   VECTOR_DB_URL - Vector DB endpoint
 *   GRAPHITI_ADAPTER_URL - Graphiti adapter endpoint
 *   KARATECLUB_ADAPTER_URL - KarateClub adapter endpoint
 *   COORDINATION_TIMEOUT_MS - Coordination timeout
 */
import { OpenEvolveAdapter } from './adapter';
export interface AdapterEndpoint {
    name: string;
    type: 'prover' | 'assistant' | 'retrieval' | 'database' | 'graph' | 'ml';
    url: string;
    enabled: boolean;
}
export interface CoordinationRequest {
    problem_type: string;
    domain: string;
    capabilities: string[];
    priority: 'low' | 'medium' | 'high';
    timeout_ms?: number;
}
export interface CoordinationResult {
    adapter_name: string;
    adapter_type: string;
    status: 'success' | 'failure' | 'partial';
    result?: any;
    error?: string;
    latency_ms: number;
}
export interface CoordinationPlan {
    selected_adapters: AdapterEndpoint[];
    fallback_order: string[];
    parallel_execution: boolean;
    estimated_duration_ms: number;
}
export declare class IntegrationCoordinator {
    private readonly openEvolveAdapter;
    private readonly timeout_ms;
    private readonly logger;
    private readonly correlationId;
    private readonly adapters;
    private readonly httpClient;
    constructor(openEvolveAdapter: OpenEvolveAdapter, timeout_ms: number);
    private registerAdapters;
    planCoordination(request: CoordinationRequest): Promise<CoordinationPlan>;
    private selectAdapters;
    private determineFallbackOrder;
    private shouldExecuteInParallel;
    private estimateDuration;
    executeCoordination(plan: CoordinationPlan, request: CoordinationRequest): Promise<CoordinationResult[]>;
    private executeAdapter;
    getAdapterHealth(): Promise<Map<string, {
        status: string;
        latency_ms: number;
    }>>;
    private generateCorrelationId;
    getRegisteredAdapters(): AdapterEndpoint[];
    getAdapterByName(name: string): AdapterEndpoint | undefined;
}
export declare function createIntegrationCoordinator(openEvolveAdapter: OpenEvolveAdapter, timeout_ms?: number): IntegrationCoordinator;
//# sourceMappingURL=integration-coordinator.d.ts.map