/**
 * Unified Knowledge Query Interface
 *
 * Federation Constitution Compliant Multi-System Knowledge Query Engine
 *
 * @package @openevolve/unified-knowledge-query
 */
export { UnifiedKnowledgeQueryEngine, defaultEngine, EngineOptions, EngineConfig, } from './engine';
export { QueryRouter, queryRouter, } from './query-router';
export { ResultFusion, resultFusion, } from './result-fusion';
export { FallbackStrategy, fallbackStrategy, } from './fallback-strategy';
export { RAGBitsClient, GraphitiClient, VectorDBClient, } from './clients';
export { KnowledgeDomainSchema, KnowledgeTypeSchema, TemporalFilterSchema, QueryTypeSchema, UnifiedKnowledgeQuerySchema, SystemSourceSchema, KnowledgeItemSchema, EntitySchema, RelationshipSchema, SourceMetadataSchema, ConflictReportSchema, UnifiedQueryResultSchema, QueryPlanSchema, CostEstimateSchema, SystemConfigSchema, HealthStatusSchema, SystemHealthSchema, EngineMetricsSchema, QueryOptionsSchema, KnowledgeDomain, KnowledgeType, TemporalFilter, QueryType, UnifiedKnowledgeQuery, SystemSource, KnowledgeItem, Entity, Relationship, SourceMetadata, ConflictReport, UnifiedQueryResult, QueryPlan, CostEstimate, SystemConfig, HealthStatus, SystemHealth, EngineMetrics, QueryOptions, validateQuery, validateResult, isValidQuery, isValidResult, } from './canonical';
export { Logger, LogLevel, CircuitBreaker, CircuitState, CircuitBreakerOptions, CircuitBreakerStats, } from '@openevolve/glue-lib';
//# sourceMappingURL=index.d.ts.map