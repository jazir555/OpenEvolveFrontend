/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Memory Canonical Schemas
 *
 * Canonical data models for ICR Contextual Mode Memory integration with Graphiti.
 * These schemas define the Anti-Corruption Layer (ACL) contract for memory operations.
 * All memory data MUST conform to these schemas before storage/retrieval.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Air Gap: No imports from core-projects
 * - Runtime Truth: Schemas reflect actual memory patterns
 * - Configuration Explicitness: All fields required (no magic defaults)
 * - UTC: All timestamps in UTC ISO-8601 format
 * - Idempotency: Memory operations safe to replay
 */
import { z } from 'zod';
/**
 * Refinement outcome enumeration
 */
export declare const RefinementOutcomeSchema: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
export type RefinementOutcome = z.infer<typeof RefinementOutcomeSchema>;
/**
 * Pattern type enumeration
 */
export declare const PatternTypeSchema: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
export type PatternType = z.infer<typeof PatternTypeSchema>;
/**
 * Memory metadata included in all memory operations
 */
export declare const MemoryMetadataSchema: z.ZodObject<{
    correlation_id: z.ZodString;
    timestamp_utc: z.ZodString;
    source_service: z.ZodDefault<z.ZodString>;
    session_id: z.ZodString;
}, "strip", z.ZodTypeAny, {
    correlation_id: string;
    source_service: string;
    timestamp_utc: string;
    session_id: string;
}, {
    correlation_id: string;
    timestamp_utc: string;
    session_id: string;
    source_service?: string | undefined;
}>;
export type MemoryMetadata = z.infer<typeof MemoryMetadataSchema>;
/**
 * Refinement Memory Schema
 * Captures insights from individual refinement iterations
 */
export declare const RefinementMemorySchema: z.ZodObject<{
    session_id: z.ZodString;
    iteration_number: z.ZodNumber;
    refinement_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
    prompt: z.ZodString;
    content: z.ZodString;
    outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
    insights: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    suggested_features: z.ZodOptional<z.ZodString>;
    bug_fixes: z.ZodOptional<z.ZodString>;
    quality_metrics: z.ZodOptional<z.ZodObject<{
        novelty_score: z.ZodOptional<z.ZodNumber>;
        quality_score: z.ZodOptional<z.ZodNumber>;
        improvement_percentage: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        quality_score?: number | undefined;
        novelty_score?: number | undefined;
        improvement_percentage?: number | undefined;
    }, {
        quality_score?: number | undefined;
        novelty_score?: number | undefined;
        improvement_percentage?: number | undefined;
    }>>;
    execution_time_ms: z.ZodNumber;
    timestamp_utc: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    content: string;
    insights: string[];
    refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
    prompt: string;
    timestamp_utc: string;
    execution_time_ms: number;
    iteration_number: number;
    session_id: string;
    outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    metadata?: Record<string, any> | undefined;
    suggested_features?: string | undefined;
    bug_fixes?: string | undefined;
    quality_metrics?: {
        quality_score?: number | undefined;
        novelty_score?: number | undefined;
        improvement_percentage?: number | undefined;
    } | undefined;
}, {
    content: string;
    refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
    prompt: string;
    timestamp_utc: string;
    execution_time_ms: number;
    iteration_number: number;
    session_id: string;
    outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    metadata?: Record<string, any> | undefined;
    insights?: string[] | undefined;
    suggested_features?: string | undefined;
    bug_fixes?: string | undefined;
    quality_metrics?: {
        quality_score?: number | undefined;
        novelty_score?: number | undefined;
        improvement_percentage?: number | undefined;
    } | undefined;
}>;
export type RefinementMemory = z.infer<typeof RefinementMemorySchema>;
/**
 * Batch refinement memories for session-level storage
 */
export declare const RefinementInsightsSchema: z.ZodObject<{
    session_id: z.ZodString;
    mode: z.ZodEnum<["refine", "contextual", "agentic", "deepthink"]>;
    iterations: z.ZodArray<z.ZodObject<{
        session_id: z.ZodString;
        iteration_number: z.ZodNumber;
        refinement_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
        prompt: z.ZodString;
        content: z.ZodString;
        outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
        insights: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        suggested_features: z.ZodOptional<z.ZodString>;
        bug_fixes: z.ZodOptional<z.ZodString>;
        quality_metrics: z.ZodOptional<z.ZodObject<{
            novelty_score: z.ZodOptional<z.ZodNumber>;
            quality_score: z.ZodOptional<z.ZodNumber>;
            improvement_percentage: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            quality_score?: number | undefined;
            novelty_score?: number | undefined;
            improvement_percentage?: number | undefined;
        }, {
            quality_score?: number | undefined;
            novelty_score?: number | undefined;
            improvement_percentage?: number | undefined;
        }>>;
        execution_time_ms: z.ZodNumber;
        timestamp_utc: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        insights: string[];
        refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        prompt: string;
        timestamp_utc: string;
        execution_time_ms: number;
        iteration_number: number;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        metadata?: Record<string, any> | undefined;
        suggested_features?: string | undefined;
        bug_fixes?: string | undefined;
        quality_metrics?: {
            quality_score?: number | undefined;
            novelty_score?: number | undefined;
            improvement_percentage?: number | undefined;
        } | undefined;
    }, {
        content: string;
        refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        prompt: string;
        timestamp_utc: string;
        execution_time_ms: number;
        iteration_number: number;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        metadata?: Record<string, any> | undefined;
        insights?: string[] | undefined;
        suggested_features?: string | undefined;
        bug_fixes?: string | undefined;
        quality_metrics?: {
            quality_score?: number | undefined;
            novelty_score?: number | undefined;
            improvement_percentage?: number | undefined;
        } | undefined;
    }>, "many">;
    total_iterations: z.ZodNumber;
    successful_iterations: z.ZodNumber;
    failed_iterations: z.ZodNumber;
    total_execution_time_ms: z.ZodNumber;
    average_quality_score: z.ZodOptional<z.ZodNumber>;
    overall_outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
    key_patterns_discovered: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    lessons_learned: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    session_start_utc: z.ZodString;
    session_end_utc: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    mode: "refine" | "deepthink" | "agentic" | "contextual";
    iterations: {
        content: string;
        insights: string[];
        refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        prompt: string;
        timestamp_utc: string;
        execution_time_ms: number;
        iteration_number: number;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        metadata?: Record<string, any> | undefined;
        suggested_features?: string | undefined;
        bug_fixes?: string | undefined;
        quality_metrics?: {
            quality_score?: number | undefined;
            novelty_score?: number | undefined;
            improvement_percentage?: number | undefined;
        } | undefined;
    }[];
    session_id: string;
    total_iterations: number;
    successful_iterations: number;
    failed_iterations: number;
    total_execution_time_ms: number;
    overall_outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    key_patterns_discovered: string[];
    lessons_learned: string[];
    session_start_utc: string;
    session_end_utc: string;
    metadata?: Record<string, any> | undefined;
    average_quality_score?: number | undefined;
}, {
    mode: "refine" | "deepthink" | "agentic" | "contextual";
    iterations: {
        content: string;
        refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        prompt: string;
        timestamp_utc: string;
        execution_time_ms: number;
        iteration_number: number;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        metadata?: Record<string, any> | undefined;
        insights?: string[] | undefined;
        suggested_features?: string | undefined;
        bug_fixes?: string | undefined;
        quality_metrics?: {
            quality_score?: number | undefined;
            novelty_score?: number | undefined;
            improvement_percentage?: number | undefined;
        } | undefined;
    }[];
    session_id: string;
    total_iterations: number;
    successful_iterations: number;
    failed_iterations: number;
    total_execution_time_ms: number;
    overall_outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    session_start_utc: string;
    session_end_utc: string;
    metadata?: Record<string, any> | undefined;
    average_quality_score?: number | undefined;
    key_patterns_discovered?: string[] | undefined;
    lessons_learned?: string[] | undefined;
}>;
export type RefinementInsights = z.infer<typeof RefinementInsightsSchema>;
/**
 * Agent interaction type
 */
export declare const AgentTypeSchema: z.ZodEnum<["main_generator", "iterative_agent", "memory_agent", "quality_agent", "custom_agent"]>;
export type AgentType = z.infer<typeof AgentTypeSchema>;
/**
 * Agent interaction record
 */
export declare const AgentInteractionSchema: z.ZodObject<{
    agent_type: z.ZodEnum<["main_generator", "iterative_agent", "memory_agent", "quality_agent", "custom_agent"]>;
    agent_name: z.ZodOptional<z.ZodString>;
    content: z.ZodString;
    timestamp_utc: z.ZodString;
    execution_time_ms: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    content: string;
    timestamp_utc: string;
    agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
    metadata?: Record<string, any> | undefined;
    execution_time_ms?: number | undefined;
    agent_name?: string | undefined;
}, {
    content: string;
    timestamp_utc: string;
    agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
    metadata?: Record<string, any> | undefined;
    execution_time_ms?: number | undefined;
    agent_name?: string | undefined;
}>;
export type AgentInteraction = z.infer<typeof AgentInteractionSchema>;
/**
 * Contextual Session Schema
 * Captures full context of a contextual mode session
 */
export declare const ContextualSessionSchema: z.ZodObject<{
    session_id: z.ZodString;
    mode: z.ZodLiteral<"contextual">;
    prompt: z.ZodString;
    agents_involved: z.ZodArray<z.ZodEnum<["main_generator", "iterative_agent", "memory_agent", "quality_agent", "custom_agent"]>, "many">;
    interactions: z.ZodArray<z.ZodObject<{
        agent_type: z.ZodEnum<["main_generator", "iterative_agent", "memory_agent", "quality_agent", "custom_agent"]>;
        agent_name: z.ZodOptional<z.ZodString>;
        content: z.ZodString;
        timestamp_utc: z.ZodString;
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        timestamp_utc: string;
        agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
        metadata?: Record<string, any> | undefined;
        execution_time_ms?: number | undefined;
        agent_name?: string | undefined;
    }, {
        content: string;
        timestamp_utc: string;
        agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
        metadata?: Record<string, any> | undefined;
        execution_time_ms?: number | undefined;
        agent_name?: string | undefined;
    }>, "many">;
    context_window: z.ZodOptional<z.ZodNumber>;
    memory_compression_events: z.ZodOptional<z.ZodArray<z.ZodObject<{
        timestamp_utc: z.ZodString;
        compressed_message_count: z.ZodNumber;
        compression_ratio: z.ZodNumber;
        bytes_saved: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        timestamp_utc: string;
        compressed_message_count: number;
        compression_ratio: number;
        bytes_saved: number;
    }, {
        timestamp_utc: string;
        compressed_message_count: number;
        compression_ratio: number;
        bytes_saved: number;
    }>, "many">>;
    successes: z.ZodDefault<z.ZodNumber>;
    failures: z.ZodDefault<z.ZodNumber>;
    duration_ms: z.ZodNumber;
    start_time_utc: z.ZodString;
    end_time_utc: z.ZodString;
    final_output: z.ZodOptional<z.ZodString>;
    quality_score: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    duration_ms: number;
    prompt: string;
    mode: "contextual";
    session_id: string;
    agents_involved: ("main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent")[];
    interactions: {
        content: string;
        timestamp_utc: string;
        agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
        metadata?: Record<string, any> | undefined;
        execution_time_ms?: number | undefined;
        agent_name?: string | undefined;
    }[];
    successes: number;
    failures: number;
    start_time_utc: string;
    end_time_utc: string;
    metadata?: Record<string, any> | undefined;
    quality_score?: number | undefined;
    memory_compression_events?: {
        timestamp_utc: string;
        compressed_message_count: number;
        compression_ratio: number;
        bytes_saved: number;
    }[] | undefined;
    context_window?: number | undefined;
    final_output?: string | undefined;
}, {
    duration_ms: number;
    prompt: string;
    mode: "contextual";
    session_id: string;
    agents_involved: ("main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent")[];
    interactions: {
        content: string;
        timestamp_utc: string;
        agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
        metadata?: Record<string, any> | undefined;
        execution_time_ms?: number | undefined;
        agent_name?: string | undefined;
    }[];
    start_time_utc: string;
    end_time_utc: string;
    metadata?: Record<string, any> | undefined;
    quality_score?: number | undefined;
    memory_compression_events?: {
        timestamp_utc: string;
        compressed_message_count: number;
        compression_ratio: number;
        bytes_saved: number;
    }[] | undefined;
    context_window?: number | undefined;
    successes?: number | undefined;
    failures?: number | undefined;
    final_output?: string | undefined;
}>;
export type ContextualSession = z.infer<typeof ContextualSessionSchema>;
/**
 * Pattern Relationship Schema
 * Tracks relationships between refinement patterns across sessions
 */
export declare const PatternRelationshipSchema: z.ZodObject<{
    pattern_id: z.ZodString;
    pattern_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
    pattern_name: z.ZodString;
    description: z.ZodString;
    related_sessions: z.ZodArray<z.ZodString, "many">;
    success_rate: z.ZodNumber;
    avg_improvement: z.ZodOptional<z.ZodNumber>;
    avg_execution_time_ms: z.ZodNumber;
    frequency: z.ZodNumber;
    last_seen_utc: z.ZodString;
    first_seen_utc: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    success_rate: number;
    description: string;
    pattern_id: string;
    pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
    pattern_name: string;
    related_sessions: string[];
    avg_execution_time_ms: number;
    frequency: number;
    last_seen_utc: string;
    first_seen_utc: string;
    metadata?: Record<string, any> | undefined;
    avg_improvement?: number | undefined;
}, {
    success_rate: number;
    description: string;
    pattern_id: string;
    pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
    pattern_name: string;
    related_sessions: string[];
    avg_execution_time_ms: number;
    frequency: number;
    last_seen_utc: string;
    first_seen_utc: string;
    metadata?: Record<string, any> | undefined;
    avg_improvement?: number | undefined;
}>;
export type PatternRelationship = z.infer<typeof PatternRelationshipSchema>;
/**
 * Memory Query Schema
 * For querying historical knowledge from Graphiti
 */
export declare const MemoryQuerySchema: z.ZodObject<{
    query: z.ZodString;
    session_context: z.ZodOptional<z.ZodString>;
    pattern_type: z.ZodOptional<z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>>;
    time_range: z.ZodOptional<z.ZodObject<{
        start_utc: z.ZodString;
        end_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        start_utc: string;
        end_utc: string;
    }, {
        start_utc: string;
        end_utc: string;
    }>>;
    min_success_rate: z.ZodOptional<z.ZodNumber>;
    max_results: z.ZodDefault<z.ZodNumber>;
    include_failed: z.ZodDefault<z.ZodBoolean>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    query: string;
    max_results: number;
    include_failed: boolean;
    correlation_id?: string | undefined;
    time_range?: {
        start_utc: string;
        end_utc: string;
    } | undefined;
    pattern_type?: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation" | undefined;
    session_context?: string | undefined;
    min_success_rate?: number | undefined;
}, {
    query: string;
    correlation_id?: string | undefined;
    time_range?: {
        start_utc: string;
        end_utc: string;
    } | undefined;
    pattern_type?: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation" | undefined;
    session_context?: string | undefined;
    min_success_rate?: number | undefined;
    max_results?: number | undefined;
    include_failed?: boolean | undefined;
}>;
export type MemoryQuery = z.infer<typeof MemoryQuerySchema>;
/**
 * Historical Knowledge Result Schema
 * Returned from memory queries
 */
export declare const HistoricalKnowledgeSchema: z.ZodObject<{
    session_id: z.ZodString;
    prompt: z.ZodString;
    pattern_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
    outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
    insights: z.ZodArray<z.ZodString, "many">;
    quality_score: z.ZodOptional<z.ZodNumber>;
    timestamp_utc: z.ZodString;
    relevance_score: z.ZodNumber;
    applicable_patterns: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    insights: string[];
    prompt: string;
    timestamp_utc: string;
    session_id: string;
    outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
    relevance_score: number;
    applicable_patterns: string[];
    metadata?: Record<string, any> | undefined;
    quality_score?: number | undefined;
}, {
    insights: string[];
    prompt: string;
    timestamp_utc: string;
    session_id: string;
    outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
    relevance_score: number;
    metadata?: Record<string, any> | undefined;
    quality_score?: number | undefined;
    applicable_patterns?: string[] | undefined;
}>;
export type HistoricalKnowledge = z.infer<typeof HistoricalKnowledgeSchema>;
/**
 * Enriched Context Schema
 * Returned when retrieving historical context for a request
 */
export declare const EnrichedContextSchema: z.ZodObject<{
    query: z.ZodString;
    historical_knowledge: z.ZodArray<z.ZodObject<{
        session_id: z.ZodString;
        prompt: z.ZodString;
        pattern_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
        outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
        insights: z.ZodArray<z.ZodString, "many">;
        quality_score: z.ZodOptional<z.ZodNumber>;
        timestamp_utc: z.ZodString;
        relevance_score: z.ZodNumber;
        applicable_patterns: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        applicable_patterns: string[];
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
    }, {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        applicable_patterns?: string[] | undefined;
    }>, "many">;
    related_patterns: z.ZodArray<z.ZodObject<{
        pattern_id: z.ZodString;
        pattern_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
        pattern_name: z.ZodString;
        description: z.ZodString;
        related_sessions: z.ZodArray<z.ZodString, "many">;
        success_rate: z.ZodNumber;
        avg_improvement: z.ZodOptional<z.ZodNumber>;
        avg_execution_time_ms: z.ZodNumber;
        frequency: z.ZodNumber;
        last_seen_utc: z.ZodString;
        first_seen_utc: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }, {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }>, "many">;
    suggested_approaches: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    common_pitfalls: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    success_probability: z.ZodOptional<z.ZodNumber>;
    confidence_score: z.ZodNumber;
    processing_time_ms: z.ZodNumber;
    correlation_id: z.ZodString;
    timestamp_utc: z.ZodString;
}, "strip", z.ZodTypeAny, {
    correlation_id: string;
    query: string;
    timestamp_utc: string;
    historical_knowledge: {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        applicable_patterns: string[];
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
    }[];
    related_patterns: {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }[];
    suggested_approaches: string[];
    common_pitfalls: string[];
    confidence_score: number;
    processing_time_ms: number;
    success_probability?: number | undefined;
}, {
    correlation_id: string;
    query: string;
    timestamp_utc: string;
    historical_knowledge: {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        applicable_patterns?: string[] | undefined;
    }[];
    related_patterns: {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }[];
    confidence_score: number;
    processing_time_ms: number;
    suggested_approaches?: string[] | undefined;
    common_pitfalls?: string[] | undefined;
    success_probability?: number | undefined;
}>;
export type EnrichedContext = z.infer<typeof EnrichedContextSchema>;
/**
 * Memory Graph Node Schema
 */
export declare const MemoryGraphNodeSchema: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["session", "pattern", "insight", "agent", "entity"]>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
    updated_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    type: "pattern" | "agent" | "session" | "insight" | "entity";
    id: string;
    created_at: string;
    attributes: Record<string, any>;
    description?: string | undefined;
    updated_at?: string | undefined;
}, {
    name: string;
    type: "pattern" | "agent" | "session" | "insight" | "entity";
    id: string;
    created_at: string;
    description?: string | undefined;
    updated_at?: string | undefined;
    attributes?: Record<string, any> | undefined;
}>;
export type MemoryGraphNode = z.infer<typeof MemoryGraphNodeSchema>;
/**
 * Memory Graph Edge Schema
 */
export declare const MemoryGraphEdgeSchema: z.ZodObject<{
    id: z.ZodString;
    source_id: z.ZodString;
    target_id: z.ZodString;
    relationship_type: z.ZodString;
    weight: z.ZodOptional<z.ZodNumber>;
    strength: z.ZodOptional<z.ZodNumber>;
    attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
}, "strip", z.ZodTypeAny, {
    id: string;
    created_at: string;
    attributes: Record<string, any>;
    source_id: string;
    target_id: string;
    relationship_type: string;
    weight?: number | undefined;
    strength?: number | undefined;
}, {
    id: string;
    created_at: string;
    source_id: string;
    target_id: string;
    relationship_type: string;
    weight?: number | undefined;
    attributes?: Record<string, any> | undefined;
    strength?: number | undefined;
}>;
export type MemoryGraphEdge = z.infer<typeof MemoryGraphEdgeSchema>;
/**
 * Memory Graph Schema
 * Represents the contextual knowledge graph
 */
export declare const MemoryGraphSchema: z.ZodObject<{
    nodes: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["session", "pattern", "insight", "agent", "entity"]>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodAny>>;
        created_at: z.ZodString;
        updated_at: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        type: "pattern" | "agent" | "session" | "insight" | "entity";
        id: string;
        created_at: string;
        attributes: Record<string, any>;
        description?: string | undefined;
        updated_at?: string | undefined;
    }, {
        name: string;
        type: "pattern" | "agent" | "session" | "insight" | "entity";
        id: string;
        created_at: string;
        description?: string | undefined;
        updated_at?: string | undefined;
        attributes?: Record<string, any> | undefined;
    }>, "many">;
    edges: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        source_id: z.ZodString;
        target_id: z.ZodString;
        relationship_type: z.ZodString;
        weight: z.ZodOptional<z.ZodNumber>;
        strength: z.ZodOptional<z.ZodNumber>;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodAny>>;
        created_at: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        id: string;
        created_at: string;
        attributes: Record<string, any>;
        source_id: string;
        target_id: string;
        relationship_type: string;
        weight?: number | undefined;
        strength?: number | undefined;
    }, {
        id: string;
        created_at: string;
        source_id: string;
        target_id: string;
        relationship_type: string;
        weight?: number | undefined;
        attributes?: Record<string, any> | undefined;
        strength?: number | undefined;
    }>, "many">;
    session_count: z.ZodNumber;
    pattern_count: z.ZodNumber;
    last_updated: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    edges: {
        id: string;
        created_at: string;
        attributes: Record<string, any>;
        source_id: string;
        target_id: string;
        relationship_type: string;
        weight?: number | undefined;
        strength?: number | undefined;
    }[];
    nodes: {
        name: string;
        type: "pattern" | "agent" | "session" | "insight" | "entity";
        id: string;
        created_at: string;
        attributes: Record<string, any>;
        description?: string | undefined;
        updated_at?: string | undefined;
    }[];
    session_count: number;
    pattern_count: number;
    last_updated: string;
    metadata?: Record<string, any> | undefined;
}, {
    edges: {
        id: string;
        created_at: string;
        source_id: string;
        target_id: string;
        relationship_type: string;
        weight?: number | undefined;
        attributes?: Record<string, any> | undefined;
        strength?: number | undefined;
    }[];
    nodes: {
        name: string;
        type: "pattern" | "agent" | "session" | "insight" | "entity";
        id: string;
        created_at: string;
        description?: string | undefined;
        updated_at?: string | undefined;
        attributes?: Record<string, any> | undefined;
    }[];
    session_count: number;
    pattern_count: number;
    last_updated: string;
    metadata?: Record<string, any> | undefined;
}>;
export type MemoryGraph = z.infer<typeof MemoryGraphSchema>;
/**
 * Storage Result Schema
 * Returned from memory storage operations
 */
export declare const StorageResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    episode_id: z.ZodOptional<z.ZodString>;
    entities_created: z.ZodDefault<z.ZodNumber>;
    relationships_created: z.ZodDefault<z.ZodNumber>;
    processing_time_ms: z.ZodNumber;
    error: z.ZodOptional<z.ZodString>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    processing_time_ms: number;
    entities_created: number;
    relationships_created: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    episode_id?: string | undefined;
}, {
    success: boolean;
    processing_time_ms: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    episode_id?: string | undefined;
    entities_created?: number | undefined;
    relationships_created?: number | undefined;
}>;
export type StorageResult = z.infer<typeof StorageResultSchema>;
/**
 * Session Memory Schema
 * Aggregates all memory for a session
 */
export declare const SessionMemorySchema: z.ZodObject<{
    session_id: z.ZodString;
    session: z.ZodObject<{
        session_id: z.ZodString;
        mode: z.ZodLiteral<"contextual">;
        prompt: z.ZodString;
        agents_involved: z.ZodArray<z.ZodEnum<["main_generator", "iterative_agent", "memory_agent", "quality_agent", "custom_agent"]>, "many">;
        interactions: z.ZodArray<z.ZodObject<{
            agent_type: z.ZodEnum<["main_generator", "iterative_agent", "memory_agent", "quality_agent", "custom_agent"]>;
            agent_name: z.ZodOptional<z.ZodString>;
            content: z.ZodString;
            timestamp_utc: z.ZodString;
            execution_time_ms: z.ZodOptional<z.ZodNumber>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
            metadata?: Record<string, any> | undefined;
            execution_time_ms?: number | undefined;
            agent_name?: string | undefined;
        }, {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
            metadata?: Record<string, any> | undefined;
            execution_time_ms?: number | undefined;
            agent_name?: string | undefined;
        }>, "many">;
        context_window: z.ZodOptional<z.ZodNumber>;
        memory_compression_events: z.ZodOptional<z.ZodArray<z.ZodObject<{
            timestamp_utc: z.ZodString;
            compressed_message_count: z.ZodNumber;
            compression_ratio: z.ZodNumber;
            bytes_saved: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            timestamp_utc: string;
            compressed_message_count: number;
            compression_ratio: number;
            bytes_saved: number;
        }, {
            timestamp_utc: string;
            compressed_message_count: number;
            compression_ratio: number;
            bytes_saved: number;
        }>, "many">>;
        successes: z.ZodDefault<z.ZodNumber>;
        failures: z.ZodDefault<z.ZodNumber>;
        duration_ms: z.ZodNumber;
        start_time_utc: z.ZodString;
        end_time_utc: z.ZodString;
        final_output: z.ZodOptional<z.ZodString>;
        quality_score: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        duration_ms: number;
        prompt: string;
        mode: "contextual";
        session_id: string;
        agents_involved: ("main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent")[];
        interactions: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
            metadata?: Record<string, any> | undefined;
            execution_time_ms?: number | undefined;
            agent_name?: string | undefined;
        }[];
        successes: number;
        failures: number;
        start_time_utc: string;
        end_time_utc: string;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
            compression_ratio: number;
            bytes_saved: number;
        }[] | undefined;
        context_window?: number | undefined;
        final_output?: string | undefined;
    }, {
        duration_ms: number;
        prompt: string;
        mode: "contextual";
        session_id: string;
        agents_involved: ("main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent")[];
        interactions: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
            metadata?: Record<string, any> | undefined;
            execution_time_ms?: number | undefined;
            agent_name?: string | undefined;
        }[];
        start_time_utc: string;
        end_time_utc: string;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
            compression_ratio: number;
            bytes_saved: number;
        }[] | undefined;
        context_window?: number | undefined;
        successes?: number | undefined;
        failures?: number | undefined;
        final_output?: string | undefined;
    }>;
    insights: z.ZodOptional<z.ZodObject<{
        session_id: z.ZodString;
        mode: z.ZodEnum<["refine", "contextual", "agentic", "deepthink"]>;
        iterations: z.ZodArray<z.ZodObject<{
            session_id: z.ZodString;
            iteration_number: z.ZodNumber;
            refinement_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
            prompt: z.ZodString;
            content: z.ZodString;
            outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
            insights: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
            suggested_features: z.ZodOptional<z.ZodString>;
            bug_fixes: z.ZodOptional<z.ZodString>;
            quality_metrics: z.ZodOptional<z.ZodObject<{
                novelty_score: z.ZodOptional<z.ZodNumber>;
                quality_score: z.ZodOptional<z.ZodNumber>;
                improvement_percentage: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            }, {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            }>>;
            execution_time_ms: z.ZodNumber;
            timestamp_utc: z.ZodString;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            content: string;
            insights: string[];
            refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
            prompt: string;
            timestamp_utc: string;
            execution_time_ms: number;
            iteration_number: number;
            session_id: string;
            outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
            metadata?: Record<string, any> | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
            quality_metrics?: {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            } | undefined;
        }, {
            content: string;
            refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
            prompt: string;
            timestamp_utc: string;
            execution_time_ms: number;
            iteration_number: number;
            session_id: string;
            outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
            metadata?: Record<string, any> | undefined;
            insights?: string[] | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
            quality_metrics?: {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            } | undefined;
        }>, "many">;
        total_iterations: z.ZodNumber;
        successful_iterations: z.ZodNumber;
        failed_iterations: z.ZodNumber;
        total_execution_time_ms: z.ZodNumber;
        average_quality_score: z.ZodOptional<z.ZodNumber>;
        overall_outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
        key_patterns_discovered: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        lessons_learned: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        session_start_utc: z.ZodString;
        session_end_utc: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        mode: "refine" | "deepthink" | "agentic" | "contextual";
        iterations: {
            content: string;
            insights: string[];
            refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
            prompt: string;
            timestamp_utc: string;
            execution_time_ms: number;
            iteration_number: number;
            session_id: string;
            outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
            metadata?: Record<string, any> | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
            quality_metrics?: {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            } | undefined;
        }[];
        session_id: string;
        total_iterations: number;
        successful_iterations: number;
        failed_iterations: number;
        total_execution_time_ms: number;
        overall_outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        key_patterns_discovered: string[];
        lessons_learned: string[];
        session_start_utc: string;
        session_end_utc: string;
        metadata?: Record<string, any> | undefined;
        average_quality_score?: number | undefined;
    }, {
        mode: "refine" | "deepthink" | "agentic" | "contextual";
        iterations: {
            content: string;
            refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
            prompt: string;
            timestamp_utc: string;
            execution_time_ms: number;
            iteration_number: number;
            session_id: string;
            outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
            metadata?: Record<string, any> | undefined;
            insights?: string[] | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
            quality_metrics?: {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            } | undefined;
        }[];
        session_id: string;
        total_iterations: number;
        successful_iterations: number;
        failed_iterations: number;
        total_execution_time_ms: number;
        overall_outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        session_start_utc: string;
        session_end_utc: string;
        metadata?: Record<string, any> | undefined;
        average_quality_score?: number | undefined;
        key_patterns_discovered?: string[] | undefined;
        lessons_learned?: string[] | undefined;
    }>>;
    related_patterns: z.ZodDefault<z.ZodArray<z.ZodObject<{
        pattern_id: z.ZodString;
        pattern_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
        pattern_name: z.ZodString;
        description: z.ZodString;
        related_sessions: z.ZodArray<z.ZodString, "many">;
        success_rate: z.ZodNumber;
        avg_improvement: z.ZodOptional<z.ZodNumber>;
        avg_execution_time_ms: z.ZodNumber;
        frequency: z.ZodNumber;
        last_seen_utc: z.ZodString;
        first_seen_utc: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }, {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }>, "many">>;
    historical_context: z.ZodDefault<z.ZodArray<z.ZodObject<{
        session_id: z.ZodString;
        prompt: z.ZodString;
        pattern_type: z.ZodEnum<["iterative_refinement", "agent_collaboration", "memory_compression", "context_switching", "tool_usage", "error_recovery", "quality_improvement", "novelty_generation", "custom"]>;
        outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
        insights: z.ZodArray<z.ZodString, "many">;
        quality_score: z.ZodOptional<z.ZodNumber>;
        timestamp_utc: z.ZodString;
        relevance_score: z.ZodNumber;
        applicable_patterns: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        applicable_patterns: string[];
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
    }, {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        applicable_patterns?: string[] | undefined;
    }>, "many">>;
    memory_graph: z.ZodOptional<z.ZodObject<{
        nodes: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            type: z.ZodEnum<["session", "pattern", "insight", "agent", "entity"]>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodAny>>;
            created_at: z.ZodString;
            updated_at: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            type: "pattern" | "agent" | "session" | "insight" | "entity";
            id: string;
            created_at: string;
            attributes: Record<string, any>;
            description?: string | undefined;
            updated_at?: string | undefined;
        }, {
            name: string;
            type: "pattern" | "agent" | "session" | "insight" | "entity";
            id: string;
            created_at: string;
            description?: string | undefined;
            updated_at?: string | undefined;
            attributes?: Record<string, any> | undefined;
        }>, "many">;
        edges: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            source_id: z.ZodString;
            target_id: z.ZodString;
            relationship_type: z.ZodString;
            weight: z.ZodOptional<z.ZodNumber>;
            strength: z.ZodOptional<z.ZodNumber>;
            attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodAny>>;
            created_at: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: string;
            created_at: string;
            attributes: Record<string, any>;
            source_id: string;
            target_id: string;
            relationship_type: string;
            weight?: number | undefined;
            strength?: number | undefined;
        }, {
            id: string;
            created_at: string;
            source_id: string;
            target_id: string;
            relationship_type: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
            strength?: number | undefined;
        }>, "many">;
        session_count: z.ZodNumber;
        pattern_count: z.ZodNumber;
        last_updated: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        edges: {
            id: string;
            created_at: string;
            attributes: Record<string, any>;
            source_id: string;
            target_id: string;
            relationship_type: string;
            weight?: number | undefined;
            strength?: number | undefined;
        }[];
        nodes: {
            name: string;
            type: "pattern" | "agent" | "session" | "insight" | "entity";
            id: string;
            created_at: string;
            attributes: Record<string, any>;
            description?: string | undefined;
            updated_at?: string | undefined;
        }[];
        session_count: number;
        pattern_count: number;
        last_updated: string;
        metadata?: Record<string, any> | undefined;
    }, {
        edges: {
            id: string;
            created_at: string;
            source_id: string;
            target_id: string;
            relationship_type: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
            strength?: number | undefined;
        }[];
        nodes: {
            name: string;
            type: "pattern" | "agent" | "session" | "insight" | "entity";
            id: string;
            created_at: string;
            description?: string | undefined;
            updated_at?: string | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        session_count: number;
        pattern_count: number;
        last_updated: string;
        metadata?: Record<string, any> | undefined;
    }>>;
    created_at: z.ZodString;
    updated_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    created_at: string;
    session_id: string;
    related_patterns: {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }[];
    session: {
        duration_ms: number;
        prompt: string;
        mode: "contextual";
        session_id: string;
        agents_involved: ("main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent")[];
        interactions: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
            metadata?: Record<string, any> | undefined;
            execution_time_ms?: number | undefined;
            agent_name?: string | undefined;
        }[];
        successes: number;
        failures: number;
        start_time_utc: string;
        end_time_utc: string;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
            compression_ratio: number;
            bytes_saved: number;
        }[] | undefined;
        context_window?: number | undefined;
        final_output?: string | undefined;
    };
    historical_context: {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        applicable_patterns: string[];
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
    }[];
    insights?: {
        mode: "refine" | "deepthink" | "agentic" | "contextual";
        iterations: {
            content: string;
            insights: string[];
            refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
            prompt: string;
            timestamp_utc: string;
            execution_time_ms: number;
            iteration_number: number;
            session_id: string;
            outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
            metadata?: Record<string, any> | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
            quality_metrics?: {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            } | undefined;
        }[];
        session_id: string;
        total_iterations: number;
        successful_iterations: number;
        failed_iterations: number;
        total_execution_time_ms: number;
        overall_outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        key_patterns_discovered: string[];
        lessons_learned: string[];
        session_start_utc: string;
        session_end_utc: string;
        metadata?: Record<string, any> | undefined;
        average_quality_score?: number | undefined;
    } | undefined;
    updated_at?: string | undefined;
    memory_graph?: {
        edges: {
            id: string;
            created_at: string;
            attributes: Record<string, any>;
            source_id: string;
            target_id: string;
            relationship_type: string;
            weight?: number | undefined;
            strength?: number | undefined;
        }[];
        nodes: {
            name: string;
            type: "pattern" | "agent" | "session" | "insight" | "entity";
            id: string;
            created_at: string;
            attributes: Record<string, any>;
            description?: string | undefined;
            updated_at?: string | undefined;
        }[];
        session_count: number;
        pattern_count: number;
        last_updated: string;
        metadata?: Record<string, any> | undefined;
    } | undefined;
}, {
    created_at: string;
    session_id: string;
    session: {
        duration_ms: number;
        prompt: string;
        mode: "contextual";
        session_id: string;
        agents_involved: ("main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent")[];
        interactions: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent" | "quality_agent" | "custom_agent";
            metadata?: Record<string, any> | undefined;
            execution_time_ms?: number | undefined;
            agent_name?: string | undefined;
        }[];
        start_time_utc: string;
        end_time_utc: string;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
            compression_ratio: number;
            bytes_saved: number;
        }[] | undefined;
        context_window?: number | undefined;
        successes?: number | undefined;
        failures?: number | undefined;
        final_output?: string | undefined;
    };
    insights?: {
        mode: "refine" | "deepthink" | "agentic" | "contextual";
        iterations: {
            content: string;
            refinement_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
            prompt: string;
            timestamp_utc: string;
            execution_time_ms: number;
            iteration_number: number;
            session_id: string;
            outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
            metadata?: Record<string, any> | undefined;
            insights?: string[] | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
            quality_metrics?: {
                quality_score?: number | undefined;
                novelty_score?: number | undefined;
                improvement_percentage?: number | undefined;
            } | undefined;
        }[];
        session_id: string;
        total_iterations: number;
        successful_iterations: number;
        failed_iterations: number;
        total_execution_time_ms: number;
        overall_outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        session_start_utc: string;
        session_end_utc: string;
        metadata?: Record<string, any> | undefined;
        average_quality_score?: number | undefined;
        key_patterns_discovered?: string[] | undefined;
        lessons_learned?: string[] | undefined;
    } | undefined;
    updated_at?: string | undefined;
    related_patterns?: {
        success_rate: number;
        description: string;
        pattern_id: string;
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        pattern_name: string;
        related_sessions: string[];
        avg_execution_time_ms: number;
        frequency: number;
        last_seen_utc: string;
        first_seen_utc: string;
        metadata?: Record<string, any> | undefined;
        avg_improvement?: number | undefined;
    }[] | undefined;
    historical_context?: {
        insights: string[];
        prompt: string;
        timestamp_utc: string;
        session_id: string;
        outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
        pattern_type: "custom" | "iterative_refinement" | "agent_collaboration" | "memory_compression" | "context_switching" | "tool_usage" | "error_recovery" | "quality_improvement" | "novelty_generation";
        relevance_score: number;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        applicable_patterns?: string[] | undefined;
    }[] | undefined;
    memory_graph?: {
        edges: {
            id: string;
            created_at: string;
            source_id: string;
            target_id: string;
            relationship_type: string;
            weight?: number | undefined;
            attributes?: Record<string, any> | undefined;
            strength?: number | undefined;
        }[];
        nodes: {
            name: string;
            type: "pattern" | "agent" | "session" | "insight" | "entity";
            id: string;
            created_at: string;
            description?: string | undefined;
            updated_at?: string | undefined;
            attributes?: Record<string, any> | undefined;
        }[];
        session_count: number;
        pattern_count: number;
        last_updated: string;
        metadata?: Record<string, any> | undefined;
    } | undefined;
}>;
export type SessionMemory = z.infer<typeof SessionMemorySchema>;
/**
 * Learning Result Schema
 * Returned from learning operations
 */
export declare const LearningResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    patterns_learned: z.ZodDefault<z.ZodNumber>;
    patterns_updated: z.ZodDefault<z.ZodNumber>;
    new_relationships: z.ZodDefault<z.ZodNumber>;
    insights_extracted: z.ZodDefault<z.ZodNumber>;
    processing_time_ms: z.ZodNumber;
    confidence_score: z.ZodOptional<z.ZodNumber>;
    error: z.ZodOptional<z.ZodString>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    processing_time_ms: number;
    patterns_learned: number;
    patterns_updated: number;
    new_relationships: number;
    insights_extracted: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    confidence_score?: number | undefined;
}, {
    success: boolean;
    processing_time_ms: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    confidence_score?: number | undefined;
    patterns_learned?: number | undefined;
    patterns_updated?: number | undefined;
    new_relationships?: number | undefined;
    insights_extracted?: number | undefined;
}>;
export type LearningResult = z.infer<typeof LearningResultSchema>;
/**
 * Session Outcome Schema
 * Used for learning from completed sessions
 */
export declare const SessionOutcomeSchema: z.ZodObject<{
    session_id: z.ZodString;
    outcome: z.ZodEnum<["success", "partial_success", "failure", "timeout", "cancelled"]>;
    quality_score: z.ZodOptional<z.ZodNumber>;
    user_satisfaction: z.ZodOptional<z.ZodNumber>;
    iteration_count: z.ZodNumber;
    success_metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    failure_reasons: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    successful_patterns: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    problematic_patterns: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    lessons_learned: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    timestamp_utc: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp_utc: string;
    iteration_count: number;
    session_id: string;
    outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    lessons_learned: string[];
    failure_reasons: string[];
    successful_patterns: string[];
    problematic_patterns: string[];
    quality_score?: number | undefined;
    user_satisfaction?: number | undefined;
    success_metrics?: Record<string, number> | undefined;
}, {
    timestamp_utc: string;
    iteration_count: number;
    session_id: string;
    outcome: "success" | "failure" | "cancelled" | "timeout" | "partial_success";
    quality_score?: number | undefined;
    lessons_learned?: string[] | undefined;
    user_satisfaction?: number | undefined;
    success_metrics?: Record<string, number> | undefined;
    failure_reasons?: string[] | undefined;
    successful_patterns?: string[] | undefined;
    problematic_patterns?: string[] | undefined;
}>;
export type SessionOutcome = z.infer<typeof SessionOutcomeSchema>;
/**
 * Validate memory data against canonical schema
 *
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Validation result with success flag and data or errors
 */
export declare function validateMemorySchema<T extends z.ZodTypeAny>(schema: T, data: unknown): {
    success: boolean;
    data?: z.infer<T>;
    errors?: string[];
};
//# sourceMappingURL=canonical.d.ts.map