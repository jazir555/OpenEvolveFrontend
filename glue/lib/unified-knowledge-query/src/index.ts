/**
 * Unified Knowledge Query Interface
 *
 * Federation Constitution Compliant Multi-System Knowledge Query Engine
 *
 * @package @openevolve/unified-knowledge-query
 */

// Main engine
export {
  UnifiedKnowledgeQueryEngine,
  defaultEngine,
  EngineOptions,
  EngineConfig,
} from './engine';

// Query router
export {
  QueryRouter,
  queryRouter,
} from './query-router';

// Result fusion
export {
  ResultFusion,
  resultFusion,
} from './result-fusion';

// Fallback strategy
export {
  FallbackStrategy,
  fallbackStrategy,
} from './fallback-strategy';

// System clients
export {
  RAGBitsClient,
  GraphitiClient,
  VectorDBClient,
} from './clients';

// Canonical schemas and types
export {
  // Schemas
  KnowledgeDomainSchema,
  KnowledgeTypeSchema,
  TemporalFilterSchema,
  QueryTypeSchema,
  UnifiedKnowledgeQuerySchema,
  SystemSourceSchema,
  KnowledgeItemSchema,
  EntitySchema,
  RelationshipSchema,
  SourceMetadataSchema,
  ConflictReportSchema,
  UnifiedQueryResultSchema,
  QueryPlanSchema,
  CostEstimateSchema,
  SystemConfigSchema,
  HealthStatusSchema,
  SystemHealthSchema,
  EngineMetricsSchema,
  QueryOptionsSchema,

  // Types
  KnowledgeDomain,
  KnowledgeType,
  TemporalFilter,
  QueryType,
  UnifiedKnowledgeQuery,
  SystemSource,
  KnowledgeItem,
  Entity,
  Relationship,
  SourceMetadata,
  ConflictReport,
  UnifiedQueryResult,
  QueryPlan,
  CostEstimate,
  SystemConfig,
  HealthStatus,
  SystemHealth,
  EngineMetrics,
  QueryOptions,

  // Validation helpers
  validateQuery,
  validateResult,

  // Type guards
  isValidQuery,
  isValidResult,
} from './canonical';

// Re-exports from glue-lib
export {
  Logger,
  LogLevel,
  CircuitBreaker,
  CircuitState,
  CircuitBreakerOptions,
  CircuitBreakerStats,
} from '@openevolve/glue-lib';
