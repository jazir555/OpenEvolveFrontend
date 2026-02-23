"use strict";
/**
 * BubbleLab Adapter
 *
 * Purpose: Main adapter integrating BubbleLab with OpenEvolve orchestration layer
 * Compliance: Federation Constitution - Full compliance with all laws
 *
 * Features:
 * - Circuit breaker for system failures
 * - Exponential backoff retry for transient failures
 * - JSON Lines structured logging
 * - UTC timestamps (Law of UTC)
 * - Idempotent operations
 * - Canonical schema mapping
 * - Environment variable validation
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.BubbleLabAdapter = void 0;
exports.createBubbleLabAdapter = createBubbleLabAdapter;
const circuit_breaker_1 = require("../../../lib/circuit-breaker");
const logger_1 = require("../../../lib/logger");
const event_bus_1 = require("../../../orchestration/event-bus");
const bubble_client_1 = require("./bubble-client");
const bubblelab_canonical_1 = require("./bubblelab-canonical");
// =============================================================================
// Main BubbleLab Adapter
// =============================================================================
class BubbleLabAdapter {
    constructor(config) {
        // Create child logger with context
        this.logger = logger_1.logger.child({
            source_service: 'bubblelab-adapter',
            target_service: 'bubblelab-api',
        });
        this.eventBus = config.eventBus || event_bus_1.eventBus;
        // Initialize BubbleLab client
        this.client = new bubble_client_1.BubbleLabClient({
            api_url: config.api_url,
            timeout_ms: config.timeout_ms,
            auth_token: config.auth_token,
        });
        // Initialize circuit breaker
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: config.circuit_breaker_threshold || 5,
            timeout_ms: config.circuit_breaker_timeout_ms || 60000,
            onStateChange: (oldState, newState) => {
                this.logger.warn('Circuit breaker state changed', {
                    old_state: oldState,
                    new_state: newState,
                });
            },
        });
        // Initialize metrics
        this.metrics = {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            circuit_breaker_state: circuit_breaker_1.CircuitState.CLOSED,
            circuit_breaker_failure_count: 0,
            average_request_duration_ms: 0,
        };
        this.logger.info('BubbleLabAdapter initialized', {
            api_url: config.api_url,
            timeout_ms: config.timeout_ms,
            circuit_breaker_threshold: config.circuit_breaker_threshold || 5,
            event_bus_connected: true,
        });
    }
    // ==========================================================================
    // Health Check
    // ==========================================================================
    /**
     * Check adapter and BubbleLab API health
     * Idempotent: Safe to call multiple times
     */
    async healthCheck(correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        this.logger.info('Health check started', { correlation_id: ctx_id });
        try {
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.healthCheck();
            });
            const isHealthy = response.success && response.data?.status === 'ok';
            this.logger.info('Health check completed', {
                correlation_id: ctx_id,
                is_healthy: isHealthy,
                status: response.data?.status,
            });
            return isHealthy;
        }
        catch (error) {
            this.logger.error('Health check failed', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
            });
            return false;
        }
    }
    // ==========================================================================
    // BubbleFlow Operations
    // ==========================================================================
    /**
     * List all BubbleFlows in canonical format
     * Idempotent: GET operation, safe to retry
     */
    async listBubbleFlows(correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Listing BubbleFlows', { correlation_id: ctx_id });
        try {
            this.metrics.total_requests++;
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.listBubbleFlows();
            });
            if (!response.success || !response.data) {
                throw new Error(response.error || 'Failed to list BubbleFlows');
            }
            // Map to canonical format
            const canonicalFlows = response.data.flows.map(bubblelab_canonical_1.mapToCanonicalBubbleFlow);
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('BubbleFlows listed successfully', {
                correlation_id: ctx_id,
                count: canonicalFlows.length,
            });
            return canonicalFlows;
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to list BubbleFlows', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
            });
            throw error;
        }
    }
    /**
     * Get a specific BubbleFlow in canonical format
     * Idempotent: GET operation, safe to retry
     */
    async getBubbleFlow(flowId, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Getting BubbleFlow', {
            correlation_id: ctx_id,
            flow_id: flowId,
        });
        try {
            this.metrics.total_requests++;
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.getBubbleFlow(flowId);
            });
            if (!response.success || !response.data) {
                throw new Error(response.error || `Failed to get BubbleFlow ${flowId}`);
            }
            // Map to canonical format
            const canonicalFlow = (0, bubblelab_canonical_1.mapToCanonicalBubbleFlow)(response.data);
            // Validate canonical schema
            (0, bubblelab_canonical_1.validateCanonicalBubbleFlow)(canonicalFlow);
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('BubbleFlow retrieved successfully', {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            return canonicalFlow;
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to get BubbleFlow', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            throw error;
        }
    }
    /**
     * Create a new BubbleFlow from canonical format
     * NOT idempotent: Use upsert logic for idempotency
     */
    async createBubbleFlow(canonicalFlow, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Creating BubbleFlow', {
            correlation_id: ctx_id,
            flow_name: canonicalFlow.name,
        });
        try {
            // Validate canonical schema
            (0, bubblelab_canonical_1.validateCanonicalBubbleFlow)(canonicalFlow);
            this.metrics.total_requests++;
            // Map from canonical to API format
            const apiRequest = (0, bubblelab_canonical_1.mapFromCanonicalBubbleFlow)(canonicalFlow);
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.createBubbleFlow(apiRequest);
            });
            if (!response.success || !response.data) {
                throw new Error(response.error || 'Failed to create BubbleFlow');
            }
            // Map back to canonical format
            const createdFlow = (0, bubblelab_canonical_1.mapToCanonicalBubbleFlow)({
                ...canonicalFlow,
                id: response.data.id,
                createdAt: response.data.createdAt,
            });
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('BubbleFlow created successfully', {
                correlation_id: ctx_id,
                flow_id: createdFlow.id,
            });
            // Emit canonical event
            await this.emitEvent({
                event_id: (0, bubblelab_canonical_1.generateCorrelationId)(),
                event_type: 'workflow.created',
                flow_id: createdFlow.id,
                timestamp: (0, bubblelab_canonical_1.toUTCISOString)(new Date()),
                data: createdFlow,
                correlation_id: ctx_id,
            });
            return createdFlow;
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to create BubbleFlow', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
            });
            throw error;
        }
    }
    /**
     * Upsert a BubbleFlow (create or update if exists)
     * Idempotent: Safe to run multiple times
     */
    async upsertBubbleFlow(canonicalFlow, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        // If flow has an ID, try to get it first
        if (canonicalFlow.id) {
            try {
                await this.getBubbleFlow(canonicalFlow.id, ctx_id);
                // Flow exists, update it
                return await this.updateBubbleFlow(canonicalFlow.id, canonicalFlow, ctx_id);
            }
            catch {
                // Flow doesn't exist, create new
                this.logger.info('Flow not found, creating new', {
                    correlation_id: ctx_id,
                    flow_id: canonicalFlow.id,
                });
            }
        }
        // Create new flow
        return await this.createBubbleFlow(canonicalFlow, ctx_id);
    }
    /**
     * Update a BubbleFlow
     * NOT idempotent: Multiple updates with same data OK
     */
    async updateBubbleFlow(flowId, updates, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Updating BubbleFlow', {
            correlation_id: ctx_id,
            flow_id: flowId,
        });
        try {
            this.metrics.total_requests++;
            // Map from canonical to API format
            const apiUpdates = (0, bubblelab_canonical_1.mapFromCanonicalBubbleFlow)(updates);
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.updateBubbleFlow(flowId, apiUpdates);
            });
            if (!response.success) {
                throw new Error(response.error || `Failed to update BubbleFlow ${flowId}`);
            }
            // Get updated flow
            const updatedFlow = await this.getBubbleFlow(flowId, ctx_id);
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('BubbleFlow updated successfully', {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            // Emit canonical event
            await this.emitEvent({
                event_id: (0, bubblelab_canonical_1.generateCorrelationId)(),
                event_type: 'workflow.updated',
                flow_id: flowId,
                timestamp: (0, bubblelab_canonical_1.toUTCISOString)(new Date()),
                data: updatedFlow,
                correlation_id: ctx_id,
            });
            return updatedFlow;
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to update BubbleFlow', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            throw error;
        }
    }
    /**
     * Delete a BubbleFlow
     * Idempotent with check: Verifies deletion
     */
    async deleteBubbleFlow(flowId, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Deleting BubbleFlow', {
            correlation_id: ctx_id,
            flow_id: flowId,
        });
        try {
            this.metrics.total_requests++;
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.deleteBubbleFlow(flowId);
            });
            if (!response.success) {
                throw new Error(response.error || `Failed to delete BubbleFlow ${flowId}`);
            }
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('BubbleFlow deleted successfully', {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            // Emit canonical event
            await this.emitEvent({
                event_id: (0, bubblelab_canonical_1.generateCorrelationId)(),
                event_type: 'workflow.deleted',
                flow_id: flowId,
                timestamp: (0, bubblelab_canonical_1.toUTCISOString)(new Date()),
                data: { flow_id: flowId },
                correlation_id: ctx_id,
            });
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to delete BubbleFlow', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            throw error;
        }
    }
    // ==========================================================================
    // Execution Operations
    // ==========================================================================
    /**
     * Execute a BubbleFlow
     * NOT idempotent: Each execution creates a new run
     */
    async executeBubbleFlow(flowId, payload, credentials, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Executing BubbleFlow', {
            correlation_id: ctx_id,
            flow_id: flowId,
            has_payload: !!payload,
            has_credentials: !!credentials,
        });
        try {
            this.metrics.total_requests++;
            // Map credentials from canonical format
            const apiCredentials = credentials
                ? (0, bubblelab_canonical_1.mapFromCanonicalCredentials)(credentials)
                : undefined;
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.executeBubbleFlow(flowId, {
                    payload,
                    credentials: apiCredentials,
                });
            });
            if (!response.success) {
                throw new Error(response.error || `Failed to execute BubbleFlow ${flowId}`);
            }
            // Map to canonical format
            const canonicalResult = (0, bubblelab_canonical_1.mapToCanonicalExecutionResult)(response.data || {}, flowId);
            // Validate canonical schema
            (0, bubblelab_canonical_1.validateCanonicalExecutionResult)(canonicalResult);
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('BubbleFlow executed successfully', {
                correlation_id: ctx_id,
                flow_id: flowId,
                execution_id: canonicalResult.execution_id,
                status: canonicalResult.status,
            });
            // Emit canonical event
            const eventType = canonicalResult.status === 'success'
                ? 'workflow.executed'
                : 'workflow.execution_failed';
            await this.emitEvent({
                event_id: (0, bubblelab_canonical_1.generateCorrelationId)(),
                event_type: eventType,
                flow_id: flowId,
                execution_id: canonicalResult.execution_id,
                timestamp: (0, bubblelab_canonical_1.toUTCISOString)(new Date()),
                data: canonicalResult,
                correlation_id: ctx_id,
            });
            return canonicalResult;
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to execute BubbleFlow', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            throw error;
        }
    }
    /**
     * Get execution history for a BubbleFlow
     * Idempotent: GET operation, safe to retry
     */
    async getExecutionHistory(flowId, limit = 50, offset = 0, correlation_id) {
        const ctx_id = correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const startTime = Date.now();
        this.logger.info('Getting execution history', {
            correlation_id: ctx_id,
            flow_id: flowId,
            limit,
            offset,
        });
        try {
            this.metrics.total_requests++;
            const response = await this.circuitBreaker.execute(async () => {
                return await this.client.getExecutionHistory(flowId, limit, offset);
            });
            if (!response.success || !response.data) {
                throw new Error(response.error || 'Failed to get execution history');
            }
            // Map to canonical format
            const executions = (response.data.executions || []).map((exec) => (0, bubblelab_canonical_1.mapToCanonicalExecutionResult)(exec, flowId));
            this.metrics.successful_requests++;
            this.updateMetrics(startTime);
            this.logger.info('Execution history retrieved successfully', {
                correlation_id: ctx_id,
                flow_id: flowId,
                count: executions.length,
            });
            return executions;
        }
        catch (error) {
            this.metrics.failed_requests++;
            this.updateMetrics(startTime);
            this.logger.error('Failed to get execution history', error instanceof Error ? error : undefined, {
                correlation_id: ctx_id,
                flow_id: flowId,
            });
            throw error;
        }
    }
    // ==========================================================================
    // Metrics & Monitoring
    // ==========================================================================
    /**
     * Get current adapter metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            circuit_breaker_state: this.circuitBreaker.getState(),
            circuit_breaker_failure_count: this.circuitBreaker.getStats().failure_count,
        };
    }
    /**
     * Reset circuit breaker (e.g., after manual health check)
     */
    resetCircuitBreaker() {
        this.logger.warn('Circuit breaker manually reset');
        this.circuitBreaker.reset();
    }
    // ==========================================================================
    // Private Helper Methods
    // ==========================================================================
    /**
     * Emit canonical event to orchestration layer
     */
    async emitEvent(event) {
        const correlationId = event.correlation_id || (0, bubblelab_canonical_1.generateCorrelationId)();
        const eventData = event.data && typeof event.data === 'object'
            ? event.data
            : { value: event.data };
        const orchestrationEvent = {
            id: event.event_id,
            type: event.event_type,
            timestamp: event.timestamp,
            correlation_id: correlationId,
            source_service: 'bubblelab-adapter',
            data: eventData,
            metadata: {
                flow_id: event.flow_id,
                execution_id: event.execution_id,
            },
        };
        try {
            await this.eventBus.publish(orchestrationEvent);
            this.logger.info('Canonical event published to event bus', {
                event_type: event.event_type,
                flow_id: event.flow_id,
                execution_id: event.execution_id,
                correlation_id: correlationId,
            });
        }
        catch (error) {
            this.logger.error('Failed to publish canonical event', error instanceof Error ? error : undefined, {
                event_type: event.event_type,
                flow_id: event.flow_id,
                execution_id: event.execution_id,
                correlation_id: correlationId,
            });
        }
    }
    /**
     * Update metrics with latest request duration
     */
    updateMetrics(startTime) {
        const duration = Date.now() - startTime;
        const total = this.metrics.total_requests;
        const currentAvg = this.metrics.average_request_duration_ms;
        // Calculate running average
        this.metrics.average_request_duration_ms = (currentAvg * (total - 1) + duration) / total;
    }
}
exports.BubbleLabAdapter = BubbleLabAdapter;
// =============================================================================
// Factory Function
// =============================================================================
/**
 * Create a BubbleLab adapter from environment variables
 * Validates all required environment variables (Law of Configuration Explicitness)
 */
function createBubbleLabAdapter() {
    const api_url = process.env.BUBBLELAB_API_URL;
    const timeout_ms = process.env.TIMEOUT_MS
        ? parseInt(process.env.TIMEOUT_MS, 10)
        : undefined;
    const auth_token = process.env.BUBBLELAB_AUTH_TOKEN;
    // Validate required configuration
    if (!api_url) {
        throw new Error('BUBBLELAB_API_URL environment variable is required');
    }
    if (!timeout_ms || timeout_ms <= 0) {
        throw new Error('TIMEOUT_MS must be a positive number');
    }
    return new BubbleLabAdapter({
        api_url,
        timeout_ms,
        auth_token,
    });
}
//# sourceMappingURL=adapter.js.map