"use strict";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter
 *
 * Main adapter for integrating with all 7 ICR modes.
 * Provides idiomatic methods for each mode with proper error handling
 * and Federation Constitution compliance.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Idempotency: All operations safe to retry
 * - UTC: All timestamps in UTC ISO-8601 format
 * - Observability: All operations include correlation IDs
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.icrAdapter = exports.ICRAdapter = void 0;
const uuid_1 = require("uuid");
const icr_client_1 = require("./icr-client");
const memory_agent_1 = require("./memory/memory-agent");
// ============================================================================
// REQUEST BUILDERS
// ============================================================================
/**
 * Build common metadata for requests
 */
function buildMetadata(correlationId) {
    return {
        correlation_id: correlationId || (0, uuid_1.v4)(),
        timestamp_utc: new Date().toISOString(),
        source_service: 'icr-adapter'
    };
}
class ICRAdapter {
    constructor(options) {
        this.client = options?.client || icr_client_1.icrClient;
        // Initialize memory agent if configuration is provided
        if (options?.memoryAgentConfig) {
            this.memoryAgent = new memory_agent_1.EnhancedICRMemoryAgent(options.memoryAgentConfig);
        }
    }
    // ========================================================================
    // REFINE MODE
    // ========================================================================
    /**
     * Create a Refine mode request
     * Mode: Traditional iterative refinements with automated feature suggestion
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (temperature, evolution_mode, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns Refine mode response
     */
    async createRefinementRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'refine',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                evolution_mode: options.evolution_mode,
                refinement_stages: options.refinement_stages
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    // ========================================================================
    // REACT MODE
    // ========================================================================
    /**
     * Create a React mode request
     * Mode: React application development with orchestrator-coordination
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (worker_count, enable_preview, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns React mode response
     */
    async createReactRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'react',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                worker_count: options.worker_count,
                enable_preview: options.enable_preview
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    // ========================================================================
    // DEEPTHINK MODE
    // ========================================================================
    /**
     * Create a Deepthink mode request
     * Mode: Complex problem-solving through strategic decomposition
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (strategy_count, sub_strategy_count, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns Deepthink mode response
     */
    async createDeepthinkRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'deepthink',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                strategy_count: options.strategy_count,
                sub_strategy_count: options.sub_strategy_count,
                hypothesis_count: options.hypothesis_count,
                enable_iterative_corrections: options.enable_iterative_corrections,
                enable_red_team: options.enable_red_team,
                red_team_aggressiveness: options.red_team_aggressiveness
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    // ========================================================================
    // ADAPTIVE DEEPTHINK MODE
    // ========================================================================
    /**
     * Create an Adaptive Deepthink mode request
     * Mode: Full deepthink mode access to an agent
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (conversation_id, enable_streaming, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns Adaptive Deepthink mode response
     */
    async createAdaptiveDeepthinkRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'adaptive_deepthink',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                conversation_id: options.conversation_id,
                enable_streaming: options.enable_streaming
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    // ========================================================================
    // AGENTIC MODE
    // ========================================================================
    /**
     * Create an Agentic mode request
     * Mode: General-purpose iterative refinement with tool-based manipulation
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (enable_diff_tools, enable_file_tools, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns Agentic mode response
     */
    async createAgenticRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'agentic',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                conversation_id: options.conversation_id,
                enable_diff_tools: options.enable_diff_tools,
                enable_file_tools: options.enable_file_tools,
                enable_web_search: options.enable_web_search
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    // ========================================================================
    // CONTEXTUAL MODE
    // ========================================================================
    /**
     * Create a Contextual mode request
     * Mode: Iterative refinement through specialized agent collaboration
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (enable_memory_agent, memory_compression_threshold, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns Contextual mode response
     */
    async createContextualRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'contextual',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                conversation_id: options.conversation_id,
                enable_memory_agent: options.enable_memory_agent,
                memory_compression_threshold: options.memory_compression_threshold
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    /**
     * Create a Contextual mode request with memory enhancement
     * Mode: Iterative refinement with historical knowledge from Graphiti
     *
     * @param prompt - The user's prompt
     * @param options - Mode options
     * @param correlationId - Optional correlation ID for tracing
     * @returns Contextual mode response enriched with historical knowledge
     */
    async createContextualRequestWithMemory(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const sessionId = (0, uuid_1.v4)();
        // Check if memory agent is available
        if (!this.memoryAgent) {
            console.warn('Memory agent not configured. Falling back to standard contextual mode.');
            return this.createContextualRequest(prompt, options, cid);
        }
        const startTime = Date.now();
        try {
            // Step 1: Retrieve historical knowledge
            const enrichedContext = await this.memoryAgent.retrieveHistoricalKnowledge(prompt, options?.context_window || 5, cid);
            // Step 2: Enrich prompt with historical knowledge
            const enrichedPrompt = this.enrichPromptWithHistory(prompt, enrichedContext);
            // Step 3: Execute contextual request with enriched prompt
            const response = await this.createContextualRequest(enrichedPrompt, {
                ...options,
                conversation_id: options?.conversation_id || sessionId
            }, cid);
            // Step 4: Extract refinement insights from response
            const refinementInsights = this.extractRefinementInsights(sessionId, response, enrichedContext);
            // Step 5: Store insights in memory
            await this.memoryAgent.storeRefinementInsights(refinementInsights, sessionId, cid);
            // Step 6: Build contextual session for learning
            const contextualSession = this.buildContextualSession(sessionId, prompt, response, enrichedContext);
            // Step 7: Store contextual session
            await this.memoryAgent.storeContextualSession(contextualSession, cid);
            // Step 8: Learn from session if enabled
            let learningResult;
            if (options?.enable_learning) {
                const outcomes = this.generateSessionOutcomes(contextualSession, response);
                learningResult = await this.memoryAgent.learnFromSession(contextualSession, outcomes, cid);
            }
            return {
                ...response,
                enriched_context: enrichedContext,
                learning_result: learningResult
            };
        }
        catch (error) {
            console.error('Error in contextual request with memory:', error);
            // Fallback to standard contextual mode on error
            return this.createContextualRequest(prompt, options, cid);
        }
    }
    // ========================================================================
    // GENERATIVE UI MODE
    // ========================================================================
    /**
     * Create a Generative UI mode request
     * Mode: Interactive UI development with user interaction capture
     *
     * @param prompt - The user's prompt
     * @param options - Mode options (enable_interaction_capture, quality_threshold, etc.)
     * @param correlationId - Optional correlation ID for tracing
     * @returns Generative UI mode response
     */
    async createGenerativeUIRequest(prompt, options, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const request = {
            mode: 'generative_ui',
            prompt,
            options: options ? {
                temperature: options.temperature,
                top_p: options.top_p,
                max_iterations: options.max_iterations,
                model_name: options.model_name,
                provider: options.provider,
                enable_interaction_capture: options.enable_interaction_capture,
                quality_threshold: options.quality_threshold,
                max_iterations: options.max_iterations
            } : undefined,
            metadata: buildMetadata(cid)
        };
        return this.client.executeMode(request, cid);
    }
    // ========================================================================
    // HEALTH CHECK
    // ========================================================================
    /**
     * Perform health check on ICR system
     *
     * @param correlationId - Optional correlation ID for tracing
     * @returns Health check response
     */
    async healthCheck(correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        return this.client.healthCheck({ correlation_id: cid }, cid);
    }
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    /**
     * Get circuit breaker state (for monitoring)
     */
    getCircuitBreakerState() {
        return this.client.getCircuitBreakerState();
    }
    /**
     * Reset circuit breaker (for recovery)
     */
    resetCircuitBreaker() {
        this.client.resetCircuitBreaker();
    }
    /**
     * Check if memory agent is configured
     */
    hasMemoryAgent() {
        return this.memoryAgent !== undefined;
    }
    // ========================================================================
    // PRIVATE HELPER METHODS FOR MEMORY INTEGRATION
    // ========================================================================
    /**
     * Enrich prompt with historical knowledge
     */
    enrichPromptWithHistory(originalPrompt, enrichedContext) {
        const sections = [
            `# Original Request`,
            originalPrompt,
            ''
        ];
        // Add historical context if available
        if (enrichedContext.historical_knowledge.length > 0) {
            sections.push(`# Historical Context`, `Based on ${enrichedContext.historical_knowledge.length} similar past refinements:`, '');
            for (const knowledge of enrichedContext.historical_knowledge.slice(0, 3)) {
                sections.push(`## Session ${knowledge.session_id.substring(0, 8)}`, `- Pattern: ${knowledge.pattern_type}`, `- Outcome: ${knowledge.outcome}`, `- Insights: ${knowledge.insights.slice(0, 2).join('; ')}`, '');
            }
        }
        // Add suggested approaches if available
        if (enrichedContext.suggested_approaches.length > 0) {
            sections.push(`# Suggested Approaches`, ...enrichedContext.suggested_approaches.map(a => `- ${a}`), '');
        }
        // Add common pitfalls if available
        if (enrichedContext.common_pitfalls.length > 0) {
            sections.push(`# Common Pitfalls to Avoid`, ...enrichedContext.common_pitfalls.map(p => `- ${p}`), '');
        }
        return sections.join('\n');
    }
    /**
     * Extract refinement insights from response
     */
    extractRefinementInsights(sessionId, response, enrichedContext) {
        const iterations = response.result.iteration_count || 1;
        const success = response.result.success;
        return {
            session_id: sessionId,
            mode: 'contextual',
            iterations: [{
                    session_id: sessionId,
                    iteration_number: 1,
                    refinement_type: 'agent_collaboration',
                    prompt: response.request.prompt,
                    content: response.result.content,
                    outcome: success ? 'success' : 'failure',
                    insights: enrichedContext.historical_knowledge.flatMap(k => k.insights),
                    execution_time_ms: response.result.execution_time_ms,
                    timestamp_utc: new Date().toISOString()
                }],
            total_iterations: iterations,
            successful_iterations: success ? 1 : 0,
            failed_iterations: success ? 0 : 1,
            total_execution_time_ms: response.result.execution_time_ms,
            average_quality_score: response.metadata.quality_score,
            overall_outcome: success ? 'success' : 'failure',
            key_patterns_discovered: enrichedContext.related_patterns.map(p => p.pattern_name),
            lessons_learned: enrichedContext.suggested_approaches,
            session_start_utc: response.request.metadata.timestamp_utc,
            session_end_utc: response.metadata.completed_at_utc,
            metadata: {
                enriched_context_quality: enrichedContext.confidence_score,
                historical_knowledge_count: enrichedContext.historical_knowledge.length
            }
        };
    }
    /**
     * Build contextual session from response
     */
    buildContextualSession(sessionId, prompt, response, enrichedContext) {
        const interactions = response.result.agent_interactions || [];
        return {
            session_id: sessionId,
            mode: 'contextual',
            prompt,
            agents_involved: interactions.map(i => i.agent_type),
            interactions: interactions,
            context_window: enrichedContext.historical_knowledge.length,
            successes: response.result.success ? 1 : 0,
            failures: response.result.success ? 0 : 1,
            duration_ms: response.result.execution_time_ms,
            start_time_utc: response.request.metadata.timestamp_utc,
            end_time_utc: response.metadata.completed_at_utc,
            final_output: response.result.content,
            quality_score: response.metadata.quality_score,
            metadata: {
                enriched_context: enrichedContext
            }
        };
    }
    /**
     * Generate session outcomes for learning
     */
    generateSessionOutcomes(session, response) {
        const outcome = {
            session_id: session.session_id,
            outcome: response.result.success ? 'success' : 'failure',
            quality_score: session.quality_score,
            user_satisfaction: session.quality_score, // Using quality as proxy
            iteration_count: response.result.iteration_count || 1,
            success_metrics: {
                execution_time_ms: session.duration_ms,
                agent_count: session.agents_involved.length,
                interaction_count: session.interactions.length
            },
            failure_reasons: response.result.success ? [] : ['Execution failed'],
            successful_patterns: session.metadata?.enriched_context?.related_patterns
                ?.map(p => p.pattern_name) || [],
            problematic_patterns: [],
            lessons_learned: session.metadata?.enriched_context?.suggested_approaches || [],
            timestamp_utc: session.end_time_utc
        };
        return [outcome];
    }
}
exports.ICRAdapter = ICRAdapter;
// ============================================================================
// SINGLETON INSTANCE
// ============================================================================
/**
 * Default ICR adapter instance
 */
exports.icrAdapter = new ICRAdapter();
//# sourceMappingURL=adapter.js.map