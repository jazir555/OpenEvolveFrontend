/**
 * OpenEvolve Integration
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: NO IMPORTS from core-projects/openevolve
 * - We integrate via HTTP API only
 * - Law of Runtime Truth: Probe OpenEvolve API before use
 *
 * This module provides integration hooks between OpenEvolve and the
 * evolved code capture system.
 */
import { Logger } from '../../logger';
import { EvolvedCodeCapturer } from './capturer';
import { CaptureResult } from './canonical';
export interface OpenEvolveIntegrationConfig {
    openevolve_api_url: string;
    openevolve_api_key?: string;
    capturer: EvolvedCodeCapturer;
    webhook_enabled: boolean;
    webhook_path: string;
    auto_capture_on_completion: boolean;
    capture_threshold_fitness?: number;
    capture_top_n_solutions?: number;
    timeout_ms?: number;
    max_retries?: number;
    logger?: Logger;
}
/**
 * OpenEvolve API Client
 *
 * Lightweight client for interacting with OpenEvolve via HTTP
 * Following Law of the Air Gap: No direct imports, only HTTP
 */
export declare class OpenEvolveClient {
    private readonly apiUrl;
    private readonly apiKey;
    private readonly logger;
    private readonly timeout;
    constructor(apiUrl: string, apiKey?: string, logger?: Logger, timeout?: number);
    /**
     * Get evolution result by ID
     */
    getEvolutionResult(evolutionId: string): Promise<any>;
    /**
     * List recent evolutions
     */
    listEvolutions(limit?: number): Promise<any[]>;
}
/**
 * OpenEvolve Integration
 *
 * Orchestrates the capture of evolved code from OpenEvolve
 */
export declare class OpenEvolveIntegration {
    private readonly config;
    private readonly logger;
    private readonly client;
    private readonly capturer;
    private initialized;
    constructor(config: OpenEvolveIntegrationConfig);
    /**
     * Initialize integration
     */
    initialize(): Promise<void>;
    /**
     * Capture evolution from OpenEvolve
     * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
     */
    captureEvolution(evolutionId: string, correlationId?: string): Promise<CaptureResult[]>;
    /**
     * Convert OpenEvolve result to canonical Problem
     */
    private convertToProblem;
    /**
     * Convert OpenEvolve result to canonical EvolvedCode
     */
    private convertToEvolvedCodes;
    /**
     * Convert single solution to canonical EvolvedCode
     */
    private convertSolutionToEvolvedCode;
    /**
     * Convert to canonical EvolutionMetrics
     */
    private convertToMetrics;
    /**
     * Map OpenEvolve problem type to canonical type
     */
    private mapProblemType;
    /**
     * Map language to canonical Language
     */
    private mapLanguage;
    /**
     * Handle webhook from OpenEvolve
     * Called when evolution completes
     */
    handleWebhook(payload: any, correlationId?: string): Promise<CaptureResult[]>;
    /**
     * Check integration health
     */
    healthCheck(): Promise<{
        healthy: boolean;
        initialized: boolean;
        openevolve_connected: boolean;
        capturer_healthy: boolean;
    }>;
    /**
     * Close integration and cleanup resources
     */
    close(): Promise<void>;
}
export type { OpenEvolveIntegrationConfig };
//# sourceMappingURL=openevolve-integration.d.ts.map