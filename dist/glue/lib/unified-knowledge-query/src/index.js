"use strict";
/**
 * Unified Knowledge Query Interface
 *
 * Federation Constitution Compliant Multi-System Knowledge Query Engine
 *
 * @package @openevolve/unified-knowledge-query
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CircuitBreakerStats = exports.CircuitBreakerOptions = exports.CircuitState = exports.CircuitBreaker = exports.LogLevel = exports.Logger = exports.isValidResult = exports.isValidQuery = exports.validateResult = exports.validateQuery = exports.QueryOptionsSchema = exports.EngineMetricsSchema = exports.SystemHealthSchema = exports.HealthStatusSchema = exports.SystemConfigSchema = exports.CostEstimateSchema = exports.QueryPlanSchema = exports.UnifiedQueryResultSchema = exports.ConflictReportSchema = exports.SourceMetadataSchema = exports.RelationshipSchema = exports.EntitySchema = exports.KnowledgeItemSchema = exports.SystemSourceSchema = exports.UnifiedKnowledgeQuerySchema = exports.QueryTypeSchema = exports.TemporalFilterSchema = exports.KnowledgeTypeSchema = exports.KnowledgeDomainSchema = exports.VectorDBClient = exports.GraphitiClient = exports.RAGBitsClient = exports.fallbackStrategy = exports.FallbackStrategy = exports.resultFusion = exports.ResultFusion = exports.queryRouter = exports.QueryRouter = exports.EngineConfig = exports.EngineOptions = exports.defaultEngine = exports.UnifiedKnowledgeQueryEngine = void 0;
// Main engine
var engine_1 = require("./engine");
Object.defineProperty(exports, "UnifiedKnowledgeQueryEngine", { enumerable: true, get: function () { return engine_1.UnifiedKnowledgeQueryEngine; } });
Object.defineProperty(exports, "defaultEngine", { enumerable: true, get: function () { return engine_1.defaultEngine; } });
Object.defineProperty(exports, "EngineOptions", { enumerable: true, get: function () { return engine_1.EngineOptions; } });
Object.defineProperty(exports, "EngineConfig", { enumerable: true, get: function () { return engine_1.EngineConfig; } });
// Query router
var query_router_1 = require("./query-router");
Object.defineProperty(exports, "QueryRouter", { enumerable: true, get: function () { return query_router_1.QueryRouter; } });
Object.defineProperty(exports, "queryRouter", { enumerable: true, get: function () { return query_router_1.queryRouter; } });
// Result fusion
var result_fusion_1 = require("./result-fusion");
Object.defineProperty(exports, "ResultFusion", { enumerable: true, get: function () { return result_fusion_1.ResultFusion; } });
Object.defineProperty(exports, "resultFusion", { enumerable: true, get: function () { return result_fusion_1.resultFusion; } });
// Fallback strategy
var fallback_strategy_1 = require("./fallback-strategy");
Object.defineProperty(exports, "FallbackStrategy", { enumerable: true, get: function () { return fallback_strategy_1.FallbackStrategy; } });
Object.defineProperty(exports, "fallbackStrategy", { enumerable: true, get: function () { return fallback_strategy_1.fallbackStrategy; } });
// System clients
var clients_1 = require("./clients");
Object.defineProperty(exports, "RAGBitsClient", { enumerable: true, get: function () { return clients_1.RAGBitsClient; } });
Object.defineProperty(exports, "GraphitiClient", { enumerable: true, get: function () { return clients_1.GraphitiClient; } });
Object.defineProperty(exports, "VectorDBClient", { enumerable: true, get: function () { return clients_1.VectorDBClient; } });
// Canonical schemas and types
var canonical_1 = require("./canonical");
// Schemas
Object.defineProperty(exports, "KnowledgeDomainSchema", { enumerable: true, get: function () { return canonical_1.KnowledgeDomainSchema; } });
Object.defineProperty(exports, "KnowledgeTypeSchema", { enumerable: true, get: function () { return canonical_1.KnowledgeTypeSchema; } });
Object.defineProperty(exports, "TemporalFilterSchema", { enumerable: true, get: function () { return canonical_1.TemporalFilterSchema; } });
Object.defineProperty(exports, "QueryTypeSchema", { enumerable: true, get: function () { return canonical_1.QueryTypeSchema; } });
Object.defineProperty(exports, "UnifiedKnowledgeQuerySchema", { enumerable: true, get: function () { return canonical_1.UnifiedKnowledgeQuerySchema; } });
Object.defineProperty(exports, "SystemSourceSchema", { enumerable: true, get: function () { return canonical_1.SystemSourceSchema; } });
Object.defineProperty(exports, "KnowledgeItemSchema", { enumerable: true, get: function () { return canonical_1.KnowledgeItemSchema; } });
Object.defineProperty(exports, "EntitySchema", { enumerable: true, get: function () { return canonical_1.EntitySchema; } });
Object.defineProperty(exports, "RelationshipSchema", { enumerable: true, get: function () { return canonical_1.RelationshipSchema; } });
Object.defineProperty(exports, "SourceMetadataSchema", { enumerable: true, get: function () { return canonical_1.SourceMetadataSchema; } });
Object.defineProperty(exports, "ConflictReportSchema", { enumerable: true, get: function () { return canonical_1.ConflictReportSchema; } });
Object.defineProperty(exports, "UnifiedQueryResultSchema", { enumerable: true, get: function () { return canonical_1.UnifiedQueryResultSchema; } });
Object.defineProperty(exports, "QueryPlanSchema", { enumerable: true, get: function () { return canonical_1.QueryPlanSchema; } });
Object.defineProperty(exports, "CostEstimateSchema", { enumerable: true, get: function () { return canonical_1.CostEstimateSchema; } });
Object.defineProperty(exports, "SystemConfigSchema", { enumerable: true, get: function () { return canonical_1.SystemConfigSchema; } });
Object.defineProperty(exports, "HealthStatusSchema", { enumerable: true, get: function () { return canonical_1.HealthStatusSchema; } });
Object.defineProperty(exports, "SystemHealthSchema", { enumerable: true, get: function () { return canonical_1.SystemHealthSchema; } });
Object.defineProperty(exports, "EngineMetricsSchema", { enumerable: true, get: function () { return canonical_1.EngineMetricsSchema; } });
Object.defineProperty(exports, "QueryOptionsSchema", { enumerable: true, get: function () { return canonical_1.QueryOptionsSchema; } });
// Validation helpers
Object.defineProperty(exports, "validateQuery", { enumerable: true, get: function () { return canonical_1.validateQuery; } });
Object.defineProperty(exports, "validateResult", { enumerable: true, get: function () { return canonical_1.validateResult; } });
// Type guards
Object.defineProperty(exports, "isValidQuery", { enumerable: true, get: function () { return canonical_1.isValidQuery; } });
Object.defineProperty(exports, "isValidResult", { enumerable: true, get: function () { return canonical_1.isValidResult; } });
// Re-exports from glue-lib
var glue_lib_1 = require("@openevolve/glue-lib");
Object.defineProperty(exports, "Logger", { enumerable: true, get: function () { return glue_lib_1.Logger; } });
Object.defineProperty(exports, "LogLevel", { enumerable: true, get: function () { return glue_lib_1.LogLevel; } });
Object.defineProperty(exports, "CircuitBreaker", { enumerable: true, get: function () { return glue_lib_1.CircuitBreaker; } });
Object.defineProperty(exports, "CircuitState", { enumerable: true, get: function () { return glue_lib_1.CircuitState; } });
Object.defineProperty(exports, "CircuitBreakerOptions", { enumerable: true, get: function () { return glue_lib_1.CircuitBreakerOptions; } });
Object.defineProperty(exports, "CircuitBreakerStats", { enumerable: true, get: function () { return glue_lib_1.CircuitBreakerStats; } });
//# sourceMappingURL=index.js.map