"use strict";
/**
 * Graphiti to RAGBits Sync
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify API calls work
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Circuit breakers and retries
 *
 * Synchronizes data from Graphiti (Temporal Knowledge Graph) to RAGBits (RAG system)
 * Enhances retrieval with knowledge graph entities
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.GraphitiToRAGBitsSync = void 0;
const uuid_1 = require("uuid");
const logger_1 = require("../../lib/logger");
const circuit_breaker_1 = require("../../lib/circuit-breaker");
const canonical_1 = require("./canonical");
// ============================================================================
// MAIN SYNC CLASS
// ============================================================================
/**
 * Graphiti to RAGBits Synchronization
 *
 * Handles one-way sync from Graphiti knowledge graph to RAGBits
 * Enhances retrieval with knowledge graph context
 */
class GraphitiToRAGBitsSync {
    constructor(config) {
        this.graphitiServiceName = 'graphiti';
        this.ragbitsServiceName = 'ragbits';
        this.config = config;
        this.logger = new logger_1.Logger('graphiti-to-ragbits-sync');
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: 5,
            timeout_ms: 60000,
            onStateChange: (oldState, newState) => {
                this.logger.warn('Circuit breaker state changed', {
                    old_state: oldState,
                    new_state: newState,
                    source_service: this.graphitiServiceName,
                    target_service: this.ragbitsServiceName,
                });
            },
        });
    }
    /**
     * Sync entity metadata to RAGBits
     *
     * @param entity - Graphiti entity to sync
     * @param correlationId - Correlation ID for tracing
     * @returns Sync result
     */
    async syncEntity(entity, correlationId) {
        const startTime = Date.now();
        const operationId = (0, uuid_1.v4)();
        this.logger.info('Starting entity sync to RAGBits', {
            correlation_id: correlationId,
            operation_id: operationId,
            entity_id: entity.id,
            entity_name: entity.name,
            source_service: this.graphitiServiceName,
            target_service: this.ragbitsServiceName,
        });
        const result = (0, canonical_1.createSyncResult)(operationId, canonical_1.SyncStatus.pending, canonical_1.SyncDirection.graphiti_to_ragbits, correlationId);
        result.operations_total = 1;
        try {
            // Extract keywords from entity
            const keywords = this.extractKeywords(entity);
            // Create boost factors for retrieval
            const boostFactor = this.createEntityBoost(entity);
            // Update RAGBits with entity metadata
            await this.circuitBreaker.execute(async () => {
                return await this.updateRAGBitsWithEntity(entity, keywords, boostFactor, correlationId);
            });
            result.status = canonical_1.SyncStatus.completed;
            result.operations_completed = 1;
            result.duration_ms = Date.now() - startTime;
            this.logger.info('Entity sync completed successfully', {
                correlation_id: correlationId,
                operation_id: operationId,
                entity_id: entity.id,
                entity_name: entity.name,
                keywords_count: keywords.length,
                duration_ms: result.duration_ms,
            });
            return result;
        }
        catch (error) {
            result.status = canonical_1.SyncStatus.failed;
            result.operations_failed = 1;
            result.duration_ms = Date.now() - startTime;
            result.errors.push({
                code: 'ENTITY_SYNC_FAILED',
                message: error instanceof Error ? error.message : 'Unknown error',
                details: { entity_id: entity.id, entity_name: entity.name },
            });
            this.logger.error('Entity sync failed', error, {
                correlation_id: correlationId,
                operation_id: operationId,
                entity_id: entity.id,
                duration_ms: result.duration_ms,
            });
            return result;
        }
    }
    /**
     * Enhance retrieval query with knowledge graph entities
     *
     * @param query - Original user query
     * @param correlationId - Correlation ID for tracing
     * @returns Enhanced query with entity context
     */
    async enhanceRetrieval(query, correlationId) {
        this.logger.info('Enhancing retrieval query with knowledge graph', {
            correlation_id: correlationId,
            query_length: query.length,
        });
        try {
            // Search for relevant entities in the knowledge graph
            const entities = await this.searchEntities(query, correlationId);
            // Create boost factors based on entities
            const boostFactors = [];
            const entitiesData = [];
            for (const entity of entities) {
                const boost = this.createEntityBoost(entity);
                boostFactors.push(boost);
                entitiesData.push({
                    id: entity.id,
                    name: entity.name,
                    labels: entity.labels,
                    boost_factor: boost.boost_value,
                });
            }
            // Enhance query with entity context
            const enhancedQuery = this.buildEnhancedQuery(query, entities);
            this.logger.info('Query enhancement completed', {
                correlation_id: correlationId,
                entities_found: entities.length,
                boost_factors_created: boostFactors.length,
            });
            return {
                original_query: query,
                enhanced_query: enhancedQuery,
                entities: entitiesData,
                boost_factors: Object.fromEntries(boostFactors.map((bf) => [bf.entity_id, bf.boost_value])),
                metadata: {
                    enhanced_at: new Date().toISOString(),
                    entity_count: entities.length,
                },
            };
        }
        catch (error) {
            this.logger.warn('Query enhancement failed, returning original query', error, {
                correlation_id: correlationId,
            });
            // Return original query if enhancement fails
            return {
                original_query: query,
                enhanced_query: query,
                entities: [],
                boost_factors: {},
                metadata: {
                    enhanced_at: new Date().toISOString(),
                    enhancement_failed: true,
                    error: error instanceof Error ? error.message : 'Unknown error',
                },
            };
        }
    }
    /**
     * Extract keywords from entity for retrieval enhancement
     *
     * @param entity - Graphiti entity
     * @returns Array of keywords
     */
    extractKeywords(entity) {
        const keywords = [];
        // Add entity name
        keywords.push(entity.name.toLowerCase());
        // Add labels
        for (const label of entity.labels) {
            keywords.push(label.toLowerCase());
        }
        // Add summary keywords
        if (entity.summary) {
            const words = entity.summary.split(/\s+/);
            for (const word of words) {
                if (word.length > 3) {
                    keywords.push(word.toLowerCase().replace(/[^a-z0-9]/g, ''));
                }
            }
        }
        // Remove duplicates
        const uniqueKeywords = Array.from(new Set(keywords));
        this.logger.debug('Extracted keywords from entity', {
            entity_id: entity.id,
            entity_name: entity.name,
            keywords_count: uniqueKeywords.length,
        });
        return uniqueKeywords;
    }
    /**
     * Create boost factor for entity
     *
     * @param entity - Graphiti entity
     * @returns Boost factor
     */
    createEntityBoost(entity) {
        // Calculate boost based on entity properties
        let boostValue = 1.0;
        // Boost based on number of labels (more specific = higher boost)
        boostValue += entity.labels.length * 0.1;
        // Boost if entity has summary (more context = higher boost)
        if (entity.summary) {
            boostValue += 0.2;
        }
        // Boost based on age (newer entities get slight boost)
        const createdAt = new Date(entity.created_at);
        const daysSinceCreation = (Date.now() - createdAt.getTime()) / (1000 * 60 * 60 * 24);
        if (daysSinceCreation < 30) {
            boostValue += 0.1;
        }
        const boost = {
            entity_id: entity.id,
            boost_value: Math.min(boostValue, 3.0), // Cap at 3.0
            reason: `Entity boost based on labels (${entity.labels.length}), summary (${entity.summary ? 'yes' : 'no'}), and age (${Math.floor(daysSinceCreation)} days)`,
            confidence: this.calculateEntityConfidence(entity),
        };
        this.logger.debug('Created boost factor for entity', {
            entity_id: entity.id,
            entity_name: entity.name,
            boost_value: boost.boost_value,
            confidence: boost.confidence,
        });
        return boost;
    }
    /**
     * Update RAGBits with entity metadata
     *
     * @param entity - Graphiti entity
     * @param keywords - Keywords for the entity
     * @param boostFactor - Boost factor for retrieval
     * @param correlationId - Correlation ID for tracing
     * @returns Promise that resolves when updated
     */
    async updateRAGBitsWithEntity(entity, keywords, boostFactor, correlationId) {
        this.logger.debug('Updating RAGBits with entity metadata', {
            correlation_id: correlationId,
            entity_id: entity.id,
            keywords_count: keywords.length,
            boost_value: boostFactor.boost_value,
        });
        // TODO: Implement actual RAGBits API call
        // This is a placeholder that demonstrates the structure
        // In production, this would call the RAGBits API to update metadata
        // Simulate API call
        await this.simulateApiCall(correlationId);
        this.logger.debug('RAGBits updated with entity metadata', {
            correlation_id: correlationId,
            entity_id: entity.id,
        });
    }
    /**
     * Search for entities relevant to query
     *
     * @param query - User query
     * @param correlationId - Correlation ID for tracing
     * @returns Array of relevant entities
     */
    async searchEntities(query, correlationId) {
        this.logger.debug('Searching for entities in knowledge graph', {
            correlation_id: correlationId,
            query_length: query.length,
        });
        // TODO: Implement actual Graphiti search API call
        // This is a placeholder that demonstrates the structure
        // In production, this would search the Graphiti knowledge graph
        // Simulate API call
        await this.simulateApiCall(correlationId);
        // Return empty array for now
        const entities = [];
        this.logger.debug('Entity search completed', {
            correlation_id: correlationId,
            entities_count: entities.length,
        });
        return entities;
    }
    /**
     * Build enhanced query with entity context
     *
     * @param originalQuery - Original user query
     * @param entities - Relevant entities
     * @returns Enhanced query
     */
    buildEnhancedQuery(originalQuery, entities) {
        if (entities.length === 0) {
            return originalQuery;
        }
        // Build context from entities
        const entityContexts = entities.map((entity) => {
            let context = `${entity.name}`;
            if (entity.summary) {
                context += `: ${entity.summary}`;
            }
            return context;
        });
        // Combine original query with entity context
        const enhancedQuery = `${originalQuery}\n\nRelevant entities:\n${entityContexts.join('\n')}`;
        this.logger.debug('Built enhanced query', {
            original_length: originalQuery.length,
            enhanced_length: enhancedQuery.length,
            entities_count: entities.length,
        });
        return enhancedQuery;
    }
    /**
     * Calculate confidence score for entity
     *
     * @param entity - Graphiti entity
     * @returns Confidence score (0-1)
     */
    calculateEntityConfidence(entity) {
        let confidence = 0.5; // Base confidence
        // Increase confidence based on labels
        confidence += Math.min(entity.labels.length * 0.1, 0.3);
        // Increase confidence if has summary
        if (entity.summary) {
            confidence += 0.2;
        }
        // Increase confidence based on attribute count
        const attributeCount = Object.keys(entity.attributes).length;
        confidence += Math.min(attributeCount * 0.05, 0.2);
        return Math.min(confidence, 1.0);
    }
    /**
     * Update retrieval strategy based on entities
     *
     * @param query - Original query
     * @param entities - Relevant entities
     * @returns Updated query strategy
     */
    updateRetrievalStrategy(query, entities) {
        this.logger.debug('Updating retrieval strategy with entities', {
            query_length: query.length,
            entities_count: entities.length,
        });
        // Build filters based on entities
        const filters = {
            include_entities: entities.map((e) => e.id),
        };
        // Add temporal filters if entities have temporal context
        const dates = entities
            .map((e) => e.created_at)
            .filter((d) => d !== undefined)
            .sort();
        if (dates.length > 0) {
            filters.temporal_range = {
                start: dates[0],
                end: dates[dates.length - 1],
            };
        }
        this.logger.debug('Retrieval strategy updated', {
            filters_count: Object.keys(filters).length,
        });
        return JSON.stringify(filters);
    }
    /**
     * Simulate API call (placeholder)
     *
     * @param correlationId - Correlation ID for tracing
     * @returns Promise that resolves after a delay
     */
    async simulateApiCall(correlationId) {
        // Simulate network delay
        await new Promise((resolve) => setTimeout(resolve, 100));
    }
    /**
     * Get circuit breaker stats
     *
     * @returns Circuit breaker statistics
     */
    getCircuitBreakerStats() {
        return this.circuitBreaker.getStats();
    }
    /**
     * Reset circuit breaker
     */
    resetCircuitBreaker() {
        this.circuitBreaker.reset();
        this.logger.info('Circuit breaker reset');
    }
}
exports.GraphitiToRAGBitsSync = GraphitiToRAGBitsSync;
// ============================================================================
// EXPORTS
// ============================================================================
exports.default = GraphitiToRAGBitsSync;
//# sourceMappingURL=graphiti-to-ragbits.js.map