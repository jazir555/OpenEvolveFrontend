/**
 * Z3 Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Z3 SMT solver interactions.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for Z3 data in the glue layer.
 * Do not pass raw Z3 API responses between services.
 */
import { z } from 'zod';
/**
 * Core Result Types from Z3
 */
export declare const Z3ResultType: z.ZodEnum<["sat", "unsat", "unknown"]>;
export type Z3ResultType = z.infer<typeof Z3ResultType>;
/**
 * Z3 Solver Request Schema
 *
 * Represents a request to solve a constraint problem using Z3.
 */
export declare const SolverRequest: z.ZodObject<{
    problem: z.ZodString;
    tactics: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    timeout_ms: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    problem: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    tactics?: string[] | undefined;
}, {
    timeout_ms: number;
    problem: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    tactics?: string[] | undefined;
}>;
export type SolverRequest = z.infer<typeof SolverRequest>;
/**
 * Z3 Solver Response Schema
 *
 * Represents the response from Z3 after solving a problem.
 */
export declare const SolverResponse: z.ZodObject<{
    result: z.ZodEnum<["sat", "unsat", "unknown"]>;
    explanation: z.ZodOptional<z.ZodString>;
    model: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    proof: z.ZodOptional<z.ZodString>;
    reason: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        solver_version: z.ZodOptional<z.ZodString>;
        solve_time_ms: z.ZodOptional<z.ZodNumber>;
        tactics_applied: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        memory_used_mb?: number | undefined;
        solver_version?: string | undefined;
        solve_time_ms?: number | undefined;
        tactics_applied?: string[] | undefined;
    }, {
        memory_used_mb?: number | undefined;
        solver_version?: string | undefined;
        solve_time_ms?: number | undefined;
        tactics_applied?: string[] | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    result: "unknown" | "sat" | "unsat";
    correlation_id?: string | undefined;
    metadata?: {
        memory_used_mb?: number | undefined;
        solver_version?: string | undefined;
        solve_time_ms?: number | undefined;
        tactics_applied?: string[] | undefined;
    } | undefined;
    proof?: string | undefined;
    model?: Record<string, any> | undefined;
    reason?: string | undefined;
    explanation?: string | undefined;
}, {
    timestamp: string;
    result: "unknown" | "sat" | "unsat";
    correlation_id?: string | undefined;
    metadata?: {
        memory_used_mb?: number | undefined;
        solver_version?: string | undefined;
        solve_time_ms?: number | undefined;
        tactics_applied?: string[] | undefined;
    } | undefined;
    proof?: string | undefined;
    model?: Record<string, any> | undefined;
    reason?: string | undefined;
    explanation?: string | undefined;
}>;
export type SolverResponse = z.infer<typeof SolverResponse>;
/**
 * Knowledge Graph Entity Schema
 *
 * Represents an entity extracted from mathematical content.
 */
export declare const Entity: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["variable", "constant", "function", "predicate", "theorem", "axiom", "definition", "other"]>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    source_location: z.ZodOptional<z.ZodObject<{
        file: z.ZodOptional<z.ZodString>;
        line_start: z.ZodOptional<z.ZodNumber>;
        line_end: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        file?: string | undefined;
        line_start?: number | undefined;
        line_end?: number | undefined;
    }, {
        file?: string | undefined;
        line_start?: number | undefined;
        line_end?: number | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    type: "function" | "theorem" | "definition" | "other" | "variable" | "constant" | "axiom" | "predicate";
    id: string;
    description?: string | undefined;
    properties?: Record<string, any> | undefined;
    source_location?: {
        file?: string | undefined;
        line_start?: number | undefined;
        line_end?: number | undefined;
    } | undefined;
}, {
    name: string;
    type: "function" | "theorem" | "definition" | "other" | "variable" | "constant" | "axiom" | "predicate";
    id: string;
    description?: string | undefined;
    properties?: Record<string, any> | undefined;
    source_location?: {
        file?: string | undefined;
        line_start?: number | undefined;
        line_end?: number | undefined;
    } | undefined;
}>;
export type Entity = z.infer<typeof Entity>;
/**
 * Knowledge Graph Relation Schema
 *
 * Represents a relationship between entities in mathematical content.
 */
export declare const Relation: z.ZodObject<{
    id: z.ZodString;
    source: z.ZodString;
    target: z.ZodString;
    type: z.ZodEnum<["depends_on", "implements", "refines", "contradicts", "implies", "equivalent_to", "instance_of", "uses", "defines", "proves", "other"]>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    type: "other" | "depends_on" | "instance_of" | "implements" | "refines" | "contradicts" | "implies" | "equivalent_to" | "uses" | "defines" | "proves";
    id: string;
    source: string;
    target: string;
    properties?: Record<string, any> | undefined;
}, {
    type: "other" | "depends_on" | "instance_of" | "implements" | "refines" | "contradicts" | "implies" | "equivalent_to" | "uses" | "defines" | "proves";
    id: string;
    source: string;
    target: string;
    properties?: Record<string, any> | undefined;
}>;
export type Relation = z.infer<typeof Relation>;
/**
 * Knowledge Graph Response Schema
 *
 * Represents a knowledge graph extracted from mathematical content.
 */
export declare const KnowledgeGraphResponse: z.ZodObject<{
    entities: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["variable", "constant", "function", "predicate", "theorem", "axiom", "definition", "other"]>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        source_location: z.ZodOptional<z.ZodObject<{
            file: z.ZodOptional<z.ZodString>;
            line_start: z.ZodOptional<z.ZodNumber>;
            line_end: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file?: string | undefined;
            line_start?: number | undefined;
            line_end?: number | undefined;
        }, {
            file?: string | undefined;
            line_start?: number | undefined;
            line_end?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        type: "function" | "theorem" | "definition" | "other" | "variable" | "constant" | "axiom" | "predicate";
        id: string;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        source_location?: {
            file?: string | undefined;
            line_start?: number | undefined;
            line_end?: number | undefined;
        } | undefined;
    }, {
        name: string;
        type: "function" | "theorem" | "definition" | "other" | "variable" | "constant" | "axiom" | "predicate";
        id: string;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        source_location?: {
            file?: string | undefined;
            line_start?: number | undefined;
            line_end?: number | undefined;
        } | undefined;
    }>, "many">;
    relations: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        source: z.ZodString;
        target: z.ZodString;
        type: z.ZodEnum<["depends_on", "implements", "refines", "contradicts", "implies", "equivalent_to", "instance_of", "uses", "defines", "proves", "other"]>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        type: "other" | "depends_on" | "instance_of" | "implements" | "refines" | "contradicts" | "implies" | "equivalent_to" | "uses" | "defines" | "proves";
        id: string;
        source: string;
        target: string;
        properties?: Record<string, any> | undefined;
    }, {
        type: "other" | "depends_on" | "instance_of" | "implements" | "refines" | "contradicts" | "implies" | "equivalent_to" | "uses" | "defines" | "proves";
        id: string;
        source: string;
        target: string;
        properties?: Record<string, any> | undefined;
    }>, "many">;
    metadata: z.ZodOptional<z.ZodObject<{
        extraction_method: z.ZodOptional<z.ZodString>;
        confidence_score: z.ZodOptional<z.ZodNumber>;
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        extraction_method?: string | undefined;
        confidence_score?: number | undefined;
        processing_time_ms?: number | undefined;
    }, {
        extraction_method?: string | undefined;
        confidence_score?: number | undefined;
        processing_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    entities: {
        name: string;
        type: "function" | "theorem" | "definition" | "other" | "variable" | "constant" | "axiom" | "predicate";
        id: string;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        source_location?: {
            file?: string | undefined;
            line_start?: number | undefined;
            line_end?: number | undefined;
        } | undefined;
    }[];
    relations: {
        type: "other" | "depends_on" | "instance_of" | "implements" | "refines" | "contradicts" | "implies" | "equivalent_to" | "uses" | "defines" | "proves";
        id: string;
        source: string;
        target: string;
        properties?: Record<string, any> | undefined;
    }[];
    correlation_id?: string | undefined;
    metadata?: {
        extraction_method?: string | undefined;
        confidence_score?: number | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
}, {
    timestamp: string;
    entities: {
        name: string;
        type: "function" | "theorem" | "definition" | "other" | "variable" | "constant" | "axiom" | "predicate";
        id: string;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        source_location?: {
            file?: string | undefined;
            line_start?: number | undefined;
            line_end?: number | undefined;
        } | undefined;
    }[];
    relations: {
        type: "other" | "depends_on" | "instance_of" | "implements" | "refines" | "contradicts" | "implies" | "equivalent_to" | "uses" | "defines" | "proves";
        id: string;
        source: string;
        target: string;
        properties?: Record<string, any> | undefined;
    }[];
    correlation_id?: string | undefined;
    metadata?: {
        extraction_method?: string | undefined;
        confidence_score?: number | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
}>;
export type KnowledgeGraphResponse = z.infer<typeof KnowledgeGraphResponse>;
/**
 * Transformation Functions
 *
 * Helper functions to convert between external formats and the canonical schema.
 */
/**
 * Transform raw Z3 API response to canonical SolverResponse
 */
export declare function transformZ3ResponseToCanonical(rawResponse: any, correlationId?: string): SolverResponse;
/**
 * Transform canonical SolverRequest to Z3 API format
 */
export declare function transformCanonicalToZ3Request(canonicalRequest: SolverRequest): any;
/**
 * Validate a SolverRequest against the schema
 */
export declare function validateSolverRequest(data: unknown): {
    success: boolean;
    data?: SolverRequest;
    errors?: string[];
};
/**
 * Validate a SolverResponse against the schema
 */
export declare function validateSolverResponse(data: unknown): {
    success: boolean;
    data?: SolverResponse;
    errors?: string[];
};
/**
 * Validate a KnowledgeGraphResponse against the schema
 */
export declare function validateKnowledgeGraphResponse(data: unknown): {
    success: boolean;
    data?: KnowledgeGraphResponse;
    errors?: string[];
};
/**
 * Example usage and validation examples
 */
export declare const Z3Examples: {
    validSolverRequest: SolverRequest;
    validSolverResponse: SolverResponse;
    validKnowledgeGraphResponse: KnowledgeGraphResponse;
};
//# sourceMappingURL=z3-canonical.d.ts.map