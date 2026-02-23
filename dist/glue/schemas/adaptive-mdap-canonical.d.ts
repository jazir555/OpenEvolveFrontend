/**
 * Adaptive MDAP Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Adaptive Multi-Domain
 * Adaptive Processing (MDAP) interactions. All adapters must normalize their
 * data to/from this format.
 */
import { z } from 'zod';
/**
 * Processing Domain Enum
 */
export declare const ProcessingDomain: z.ZodEnum<["text", "image", "audio", "video", "multimodal", "structured_data"]>;
export type ProcessingDomain = z.infer<typeof ProcessingDomain>;
/**
 * Adaptation Mode Enum
 */
export declare const AdaptationMode: z.ZodEnum<["static", "dynamic", "incremental", "continual"]>;
export type AdaptationMode = z.infer<typeof AdaptationMode>;
/**
 * MDAP Processing Request Schema
 */
export declare const AdaptiveMdapRequest: z.ZodObject<{
    task_id: z.ZodString;
    domain: z.ZodEnum<["text", "image", "audio", "video", "multimodal", "structured_data"]>;
    input_data: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodAny>, z.ZodArray<z.ZodAny, "many">]>;
    adaptation_config: z.ZodOptional<z.ZodObject<{
        mode: z.ZodOptional<z.ZodEnum<["static", "dynamic", "incremental", "continual"]>>;
        learning_rate: z.ZodOptional<z.ZodNumber>;
        batch_size: z.ZodOptional<z.ZodNumber>;
        threshold: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        threshold?: number | undefined;
        mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
        learning_rate?: number | undefined;
        batch_size?: number | undefined;
    }, {
        threshold?: number | undefined;
        mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
        learning_rate?: number | undefined;
        batch_size?: number | undefined;
    }>>;
    model_config: z.ZodOptional<z.ZodObject<{
        base_model: z.ZodOptional<z.ZodString>;
        fine_tuned: z.ZodOptional<z.ZodBoolean>;
        parameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        parameters?: Record<string, any> | undefined;
        base_model?: string | undefined;
        fine_tuned?: boolean | undefined;
    }, {
        parameters?: Record<string, any> | undefined;
        base_model?: string | undefined;
        fine_tuned?: boolean | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    input_data: string | any[] | Record<string, any>;
    domain: "text" | "audio" | "video" | "image" | "multimodal" | "structured_data";
    task_id: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    adaptation_config?: {
        threshold?: number | undefined;
        mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
        learning_rate?: number | undefined;
        batch_size?: number | undefined;
    } | undefined;
    model_config?: {
        parameters?: Record<string, any> | undefined;
        base_model?: string | undefined;
        fine_tuned?: boolean | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    input_data: string | any[] | Record<string, any>;
    domain: "text" | "audio" | "video" | "image" | "multimodal" | "structured_data";
    task_id: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    adaptation_config?: {
        threshold?: number | undefined;
        mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
        learning_rate?: number | undefined;
        batch_size?: number | undefined;
    } | undefined;
    model_config?: {
        parameters?: Record<string, any> | undefined;
        base_model?: string | undefined;
        fine_tuned?: boolean | undefined;
    } | undefined;
}>;
export type AdaptiveMdapRequest = z.infer<typeof AdaptiveMdapRequest>;
/**
 * MDAP Processing Response Schema
 */
export declare const AdaptiveMdapResponse: z.ZodObject<{
    task_id: z.ZodString;
    status: z.ZodEnum<["pending", "processing", "completed", "failed", "timeout"]>;
    result: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodAny>, z.ZodArray<z.ZodAny, "many">]>>;
    adaptations: z.ZodOptional<z.ZodObject<{
        adaptations_made: z.ZodOptional<z.ZodNumber>;
        adaptation_history: z.ZodOptional<z.ZodArray<z.ZodObject<{
            timestamp: z.ZodString;
            change_type: z.ZodString;
            performance_delta: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            timestamp: string;
            change_type: string;
            performance_delta?: number | undefined;
        }, {
            timestamp: string;
            change_type: string;
            performance_delta?: number | undefined;
        }>, "many">>;
        model_version: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        adaptations_made?: number | undefined;
        adaptation_history?: {
            timestamp: string;
            change_type: string;
            performance_delta?: number | undefined;
        }[] | undefined;
        model_version?: string | undefined;
    }, {
        adaptations_made?: number | undefined;
        adaptation_history?: {
            timestamp: string;
            change_type: string;
            performance_delta?: number | undefined;
        }[] | undefined;
        model_version?: string | undefined;
    }>>;
    performance: z.ZodOptional<z.ZodObject<{
        accuracy: z.ZodOptional<z.ZodNumber>;
        latency_ms: z.ZodOptional<z.ZodNumber>;
        throughput: z.ZodOptional<z.ZodNumber>;
        resource_usage: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        latency_ms?: number | undefined;
        resource_usage?: Record<string, number> | undefined;
        accuracy?: number | undefined;
        throughput?: number | undefined;
    }, {
        latency_ms?: number | undefined;
        resource_usage?: Record<string, number> | undefined;
        accuracy?: number | undefined;
        throughput?: number | undefined;
    }>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "processing" | "completed" | "failed" | "pending" | "timeout";
    task_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    result?: string | any[] | Record<string, any> | undefined;
    performance?: {
        latency_ms?: number | undefined;
        resource_usage?: Record<string, number> | undefined;
        accuracy?: number | undefined;
        throughput?: number | undefined;
    } | undefined;
    adaptations?: {
        adaptations_made?: number | undefined;
        adaptation_history?: {
            timestamp: string;
            change_type: string;
            performance_delta?: number | undefined;
        }[] | undefined;
        model_version?: string | undefined;
    } | undefined;
}, {
    timestamp: string;
    status: "processing" | "completed" | "failed" | "pending" | "timeout";
    task_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    result?: string | any[] | Record<string, any> | undefined;
    performance?: {
        latency_ms?: number | undefined;
        resource_usage?: Record<string, number> | undefined;
        accuracy?: number | undefined;
        throughput?: number | undefined;
    } | undefined;
    adaptations?: {
        adaptations_made?: number | undefined;
        adaptation_history?: {
            timestamp: string;
            change_type: string;
            performance_delta?: number | undefined;
        }[] | undefined;
        model_version?: string | undefined;
    } | undefined;
}>;
export type AdaptiveMdapResponse = z.infer<typeof AdaptiveMdapResponse>;
/**
 * Batch Processing Request Schema
 */
export declare const AdaptiveMdapBatchRequest: z.ZodObject<{
    batch_id: z.ZodString;
    tasks: z.ZodArray<z.ZodObject<Omit<{
        task_id: z.ZodString;
        domain: z.ZodEnum<["text", "image", "audio", "video", "multimodal", "structured_data"]>;
        input_data: z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodAny>, z.ZodArray<z.ZodAny, "many">]>;
        adaptation_config: z.ZodOptional<z.ZodObject<{
            mode: z.ZodOptional<z.ZodEnum<["static", "dynamic", "incremental", "continual"]>>;
            learning_rate: z.ZodOptional<z.ZodNumber>;
            batch_size: z.ZodOptional<z.ZodNumber>;
            threshold: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            threshold?: number | undefined;
            mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
            learning_rate?: number | undefined;
            batch_size?: number | undefined;
        }, {
            threshold?: number | undefined;
            mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
            learning_rate?: number | undefined;
            batch_size?: number | undefined;
        }>>;
        model_config: z.ZodOptional<z.ZodObject<{
            base_model: z.ZodOptional<z.ZodString>;
            fine_tuned: z.ZodOptional<z.ZodBoolean>;
            parameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            parameters?: Record<string, any> | undefined;
            base_model?: string | undefined;
            fine_tuned?: boolean | undefined;
        }, {
            parameters?: Record<string, any> | undefined;
            base_model?: string | undefined;
            fine_tuned?: boolean | undefined;
        }>>;
        timeout_ms: z.ZodNumber;
        correlation_id: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "correlation_id" | "timeout_ms">, "strip", z.ZodTypeAny, {
        input_data: string | any[] | Record<string, any>;
        domain: "text" | "audio" | "video" | "image" | "multimodal" | "structured_data";
        task_id: string;
        metadata?: Record<string, any> | undefined;
        adaptation_config?: {
            threshold?: number | undefined;
            mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
            learning_rate?: number | undefined;
            batch_size?: number | undefined;
        } | undefined;
        model_config?: {
            parameters?: Record<string, any> | undefined;
            base_model?: string | undefined;
            fine_tuned?: boolean | undefined;
        } | undefined;
    }, {
        input_data: string | any[] | Record<string, any>;
        domain: "text" | "audio" | "video" | "image" | "multimodal" | "structured_data";
        task_id: string;
        metadata?: Record<string, any> | undefined;
        adaptation_config?: {
            threshold?: number | undefined;
            mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
            learning_rate?: number | undefined;
            batch_size?: number | undefined;
        } | undefined;
        model_config?: {
            parameters?: Record<string, any> | undefined;
            base_model?: string | undefined;
            fine_tuned?: boolean | undefined;
        } | undefined;
    }>, "many">;
    config: z.ZodOptional<z.ZodObject<{
        parallelism: z.ZodOptional<z.ZodNumber>;
        stop_on_error: z.ZodOptional<z.ZodBoolean>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout_ms?: number | undefined;
        parallelism?: number | undefined;
        stop_on_error?: boolean | undefined;
    }, {
        timeout_ms?: number | undefined;
        parallelism?: number | undefined;
        stop_on_error?: boolean | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    tasks: {
        input_data: string | any[] | Record<string, any>;
        domain: "text" | "audio" | "video" | "image" | "multimodal" | "structured_data";
        task_id: string;
        metadata?: Record<string, any> | undefined;
        adaptation_config?: {
            threshold?: number | undefined;
            mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
            learning_rate?: number | undefined;
            batch_size?: number | undefined;
        } | undefined;
        model_config?: {
            parameters?: Record<string, any> | undefined;
            base_model?: string | undefined;
            fine_tuned?: boolean | undefined;
        } | undefined;
    }[];
    batch_id: string;
    correlation_id?: string | undefined;
    config?: {
        timeout_ms?: number | undefined;
        parallelism?: number | undefined;
        stop_on_error?: boolean | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
}, {
    timeout_ms: number;
    tasks: {
        input_data: string | any[] | Record<string, any>;
        domain: "text" | "audio" | "video" | "image" | "multimodal" | "structured_data";
        task_id: string;
        metadata?: Record<string, any> | undefined;
        adaptation_config?: {
            threshold?: number | undefined;
            mode?: "incremental" | "static" | "dynamic" | "continual" | undefined;
            learning_rate?: number | undefined;
            batch_size?: number | undefined;
        } | undefined;
        model_config?: {
            parameters?: Record<string, any> | undefined;
            base_model?: string | undefined;
            fine_tuned?: boolean | undefined;
        } | undefined;
    }[];
    batch_id: string;
    correlation_id?: string | undefined;
    config?: {
        timeout_ms?: number | undefined;
        parallelism?: number | undefined;
        stop_on_error?: boolean | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
}>;
export type AdaptiveMdapBatchRequest = z.infer<typeof AdaptiveMdapBatchRequest>;
/**
 * Batch Processing Response Schema
 */
export declare const AdaptiveMdapBatchResponse: z.ZodObject<{
    batch_id: z.ZodString;
    status: z.ZodEnum<["pending", "processing", "completed", "partially_completed", "failed"]>;
    results: z.ZodArray<z.ZodObject<{
        task_id: z.ZodString;
        status: z.ZodEnum<["pending", "processing", "completed", "failed", "timeout"]>;
        result: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodAny>, z.ZodArray<z.ZodAny, "many">]>>;
        adaptations: z.ZodOptional<z.ZodObject<{
            adaptations_made: z.ZodOptional<z.ZodNumber>;
            adaptation_history: z.ZodOptional<z.ZodArray<z.ZodObject<{
                timestamp: z.ZodString;
                change_type: z.ZodString;
                performance_delta: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }, {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }>, "many">>;
            model_version: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            adaptations_made?: number | undefined;
            adaptation_history?: {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }[] | undefined;
            model_version?: string | undefined;
        }, {
            adaptations_made?: number | undefined;
            adaptation_history?: {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }[] | undefined;
            model_version?: string | undefined;
        }>>;
        performance: z.ZodOptional<z.ZodObject<{
            accuracy: z.ZodOptional<z.ZodNumber>;
            latency_ms: z.ZodOptional<z.ZodNumber>;
            throughput: z.ZodOptional<z.ZodNumber>;
            resource_usage: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            latency_ms?: number | undefined;
            resource_usage?: Record<string, number> | undefined;
            accuracy?: number | undefined;
            throughput?: number | undefined;
        }, {
            latency_ms?: number | undefined;
            resource_usage?: Record<string, number> | undefined;
            accuracy?: number | undefined;
            throughput?: number | undefined;
        }>>;
        error: z.ZodOptional<z.ZodObject<{
            code: z.ZodString;
            message: z.ZodString;
            details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
        }, {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
        }>>;
        correlation_id: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        status: "processing" | "completed" | "failed" | "pending" | "timeout";
        task_id: string;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
        } | undefined;
        result?: string | any[] | Record<string, any> | undefined;
        performance?: {
            latency_ms?: number | undefined;
            resource_usage?: Record<string, number> | undefined;
            accuracy?: number | undefined;
            throughput?: number | undefined;
        } | undefined;
        adaptations?: {
            adaptations_made?: number | undefined;
            adaptation_history?: {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }[] | undefined;
            model_version?: string | undefined;
        } | undefined;
    }, {
        timestamp: string;
        status: "processing" | "completed" | "failed" | "pending" | "timeout";
        task_id: string;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
        } | undefined;
        result?: string | any[] | Record<string, any> | undefined;
        performance?: {
            latency_ms?: number | undefined;
            resource_usage?: Record<string, number> | undefined;
            accuracy?: number | undefined;
            throughput?: number | undefined;
        } | undefined;
        adaptations?: {
            adaptations_made?: number | undefined;
            adaptation_history?: {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }[] | undefined;
            model_version?: string | undefined;
        } | undefined;
    }>, "many">;
    summary: z.ZodObject<{
        total_tasks: z.ZodNumber;
        completed: z.ZodNumber;
        failed: z.ZodNumber;
        total_processing_time_ms: z.ZodOptional<z.ZodNumber>;
        average_latency_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        completed: number;
        failed: number;
        total_tasks: number;
        total_processing_time_ms?: number | undefined;
        average_latency_ms?: number | undefined;
    }, {
        completed: number;
        failed: number;
        total_tasks: number;
        total_processing_time_ms?: number | undefined;
        average_latency_ms?: number | undefined;
    }>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "processing" | "completed" | "failed" | "pending" | "partially_completed";
    results: {
        timestamp: string;
        status: "processing" | "completed" | "failed" | "pending" | "timeout";
        task_id: string;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
        } | undefined;
        result?: string | any[] | Record<string, any> | undefined;
        performance?: {
            latency_ms?: number | undefined;
            resource_usage?: Record<string, number> | undefined;
            accuracy?: number | undefined;
            throughput?: number | undefined;
        } | undefined;
        adaptations?: {
            adaptations_made?: number | undefined;
            adaptation_history?: {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }[] | undefined;
            model_version?: string | undefined;
        } | undefined;
    }[];
    summary: {
        completed: number;
        failed: number;
        total_tasks: number;
        total_processing_time_ms?: number | undefined;
        average_latency_ms?: number | undefined;
    };
    batch_id: string;
    correlation_id?: string | undefined;
}, {
    timestamp: string;
    status: "processing" | "completed" | "failed" | "pending" | "partially_completed";
    results: {
        timestamp: string;
        status: "processing" | "completed" | "failed" | "pending" | "timeout";
        task_id: string;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
        } | undefined;
        result?: string | any[] | Record<string, any> | undefined;
        performance?: {
            latency_ms?: number | undefined;
            resource_usage?: Record<string, number> | undefined;
            accuracy?: number | undefined;
            throughput?: number | undefined;
        } | undefined;
        adaptations?: {
            adaptations_made?: number | undefined;
            adaptation_history?: {
                timestamp: string;
                change_type: string;
                performance_delta?: number | undefined;
            }[] | undefined;
            model_version?: string | undefined;
        } | undefined;
    }[];
    summary: {
        completed: number;
        failed: number;
        total_tasks: number;
        total_processing_time_ms?: number | undefined;
        average_latency_ms?: number | undefined;
    };
    batch_id: string;
    correlation_id?: string | undefined;
}>;
export type AdaptiveMdapBatchResponse = z.infer<typeof AdaptiveMdapBatchResponse>;
/**
 * Error Model
 */
export declare const AdaptiveMdapError: z.ZodObject<{
    code: z.ZodEnum<["INVALID_INPUT", "DOMAIN_NOT_SUPPORTED", "MODEL_NOT_AVAILABLE", "ADAPTATION_FAILED", "PROCESSING_TIMEOUT", "RESOURCE_EXHAUSTED", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "INVALID_INPUT" | "DOMAIN_NOT_SUPPORTED" | "MODEL_NOT_AVAILABLE" | "ADAPTATION_FAILED" | "PROCESSING_TIMEOUT" | "RESOURCE_EXHAUSTED";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "INVALID_INPUT" | "DOMAIN_NOT_SUPPORTED" | "MODEL_NOT_AVAILABLE" | "ADAPTATION_FAILED" | "PROCESSING_TIMEOUT" | "RESOURCE_EXHAUSTED";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type AdaptiveMdapError = z.infer<typeof AdaptiveMdapError>;
/**
 * Validation Functions
 */
export declare function validateAdaptiveMdapRequest(data: unknown): {
    success: boolean;
    data?: AdaptiveMdapRequest;
    errors?: string[];
};
export declare function validateAdaptiveMdapResponse(data: unknown): {
    success: boolean;
    data?: AdaptiveMdapResponse;
    errors?: string[];
};
/**
 * Type Guards
 */
export declare function isAdaptiveMdapRequest(data: unknown): data is AdaptiveMdapRequest;
/**
 * Example usage
 */
export declare const AdaptiveMdapExamples: {
    validRequest: AdaptiveMdapRequest;
    validResponse: AdaptiveMdapResponse;
};
/**
 * Workflow Type Enum
 */
export declare const WorkflowType: z.ZodEnum<["evolution", "sovereign", "iterative_refinement", "formal_verification", "agent_collaboration", "knowledge_retrieval", "multi_system"]>;
export type WorkflowType = z.infer<typeof WorkflowType>;
/**
 * Complexity Dimensions Schema
 */
export declare const ComplexityDimensions: z.ZodObject<{
    text_length: z.ZodOptional<z.ZodNumber>;
    dependencies: z.ZodOptional<z.ZodNumber>;
    depth: z.ZodOptional<z.ZodNumber>;
    domain_specific: z.ZodOptional<z.ZodNumber>;
    uncertainty: z.ZodOptional<z.ZodNumber>;
    abstraction: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    dependencies?: number | undefined;
    depth?: number | undefined;
    text_length?: number | undefined;
    domain_specific?: number | undefined;
    uncertainty?: number | undefined;
    abstraction?: number | undefined;
}, {
    dependencies?: number | undefined;
    depth?: number | undefined;
    text_length?: number | undefined;
    domain_specific?: number | undefined;
    uncertainty?: number | undefined;
    abstraction?: number | undefined;
}>;
export type ComplexityDimensions = z.infer<typeof ComplexityDimensions>;
/**
 * Complexity Score Schema
 */
export declare const ComplexityScore: z.ZodObject<{
    overall_score: z.ZodNumber;
    dimensions: z.ZodOptional<z.ZodObject<{
        text_length: z.ZodOptional<z.ZodNumber>;
        dependencies: z.ZodOptional<z.ZodNumber>;
        depth: z.ZodOptional<z.ZodNumber>;
        domain_specific: z.ZodOptional<z.ZodNumber>;
        uncertainty: z.ZodOptional<z.ZodNumber>;
        abstraction: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        dependencies?: number | undefined;
        depth?: number | undefined;
        text_length?: number | undefined;
        domain_specific?: number | undefined;
        uncertainty?: number | undefined;
        abstraction?: number | undefined;
    }, {
        dependencies?: number | undefined;
        depth?: number | undefined;
        text_length?: number | undefined;
        domain_specific?: number | undefined;
        uncertainty?: number | undefined;
        abstraction?: number | undefined;
    }>>;
    confidence: z.ZodOptional<z.ZodNumber>;
    strategy: z.ZodOptional<z.ZodEnum<["DIRECT", "SEQUENTIAL", "PARALLEL", "HIERARCHICAL", "ADAPTIVE"]>>;
}, "strip", z.ZodTypeAny, {
    overall_score: number;
    confidence?: number | undefined;
    strategy?: "DIRECT" | "SEQUENTIAL" | "PARALLEL" | "HIERARCHICAL" | "ADAPTIVE" | undefined;
    dimensions?: {
        dependencies?: number | undefined;
        depth?: number | undefined;
        text_length?: number | undefined;
        domain_specific?: number | undefined;
        uncertainty?: number | undefined;
        abstraction?: number | undefined;
    } | undefined;
}, {
    overall_score: number;
    confidence?: number | undefined;
    strategy?: "DIRECT" | "SEQUENTIAL" | "PARALLEL" | "HIERARCHICAL" | "ADAPTIVE" | undefined;
    dimensions?: {
        dependencies?: number | undefined;
        depth?: number | undefined;
        text_length?: number | undefined;
        domain_specific?: number | undefined;
        uncertainty?: number | undefined;
        abstraction?: number | undefined;
    } | undefined;
}>;
export type ComplexityScore = z.infer<typeof ComplexityScore>;
/**
 * Sub-Problem Schema (for Decomposition)
 */
export declare const SubProblem: z.ZodObject<{
    id: z.ZodString;
    description: z.ZodString;
    complexity: z.ZodNumber;
    dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    estimated_effort: z.ZodOptional<z.ZodString>;
    parent_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    complexity: number;
    description: string;
    dependencies?: string[] | undefined;
    parent_id?: string | undefined;
    estimated_effort?: string | undefined;
}, {
    id: string;
    complexity: number;
    description: string;
    dependencies?: string[] | undefined;
    parent_id?: string | undefined;
    estimated_effort?: string | undefined;
}>;
export type SubProblem = z.infer<typeof SubProblem>;
/**
 * Problem Decomposition Result Schema (V2.0)
 */
export declare const ProblemDecompositionResult: z.ZodObject<{
    workflow_id: z.ZodString;
    decomposition_strategy: z.ZodEnum<["hierarchical", "functional", "domain_based", "hybrid"]>;
    sub_problems: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        description: z.ZodString;
        complexity: z.ZodNumber;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        estimated_effort: z.ZodOptional<z.ZodString>;
        parent_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        complexity: number;
        description: string;
        dependencies?: string[] | undefined;
        parent_id?: string | undefined;
        estimated_effort?: string | undefined;
    }, {
        id: string;
        complexity: number;
        description: string;
        dependencies?: string[] | undefined;
        parent_id?: string | undefined;
        estimated_effort?: string | undefined;
    }>, "many">;
    recommended_parallelization: z.ZodEnum<["none", "partial", "full"]>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    workflow_id: string;
    timestamp: string;
    sub_problems: {
        id: string;
        complexity: number;
        description: string;
        dependencies?: string[] | undefined;
        parent_id?: string | undefined;
        estimated_effort?: string | undefined;
    }[];
    decomposition_strategy: "hybrid" | "hierarchical" | "functional" | "domain_based";
    recommended_parallelization: "none" | "partial" | "full";
    metadata?: Record<string, any> | undefined;
}, {
    workflow_id: string;
    timestamp: string;
    sub_problems: {
        id: string;
        complexity: number;
        description: string;
        dependencies?: string[] | undefined;
        parent_id?: string | undefined;
        estimated_effort?: string | undefined;
    }[];
    decomposition_strategy: "hybrid" | "hierarchical" | "functional" | "domain_based";
    recommended_parallelization: "none" | "partial" | "full";
    metadata?: Record<string, any> | undefined;
}>;
export type ProblemDecompositionResult = z.infer<typeof ProblemDecompositionResult>;
/**
 * Team Member Schema
 */
export declare const TeamMember: z.ZodObject<{
    name: z.ZodString;
    role: z.ZodString;
    capabilities: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    availability: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    name: string;
    role: string;
    capabilities?: string[] | undefined;
    availability?: boolean | undefined;
}, {
    name: string;
    role: string;
    capabilities?: string[] | undefined;
    availability?: boolean | undefined;
}>;
export type TeamMember = z.infer<typeof TeamMember>;
/**
 * Team Selection Result Schema (V2.0)
 */
export declare const TeamSelectionResult: z.ZodObject<{
    workflow_id: z.ZodString;
    stage: z.ZodString;
    workflow_type: z.ZodEnum<["evolution", "sovereign", "iterative_refinement", "formal_verification", "agent_collaboration", "knowledge_retrieval", "multi_system"]>;
    complexity_score: z.ZodNumber;
    recommended_teams: z.ZodRecord<z.ZodString, z.ZodObject<{
        agents: z.ZodArray<z.ZodString, "many">;
        reasoning: z.ZodString;
        confidence: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        reasoning: string;
        agents: string[];
        confidence?: number | undefined;
    }, {
        reasoning: string;
        agents: string[];
        confidence?: number | undefined;
    }>>;
    estimated_cost: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    workflow_id: string;
    timestamp: string;
    workflow_type: "sovereign" | "evolution" | "iterative_refinement" | "agent_collaboration" | "formal_verification" | "knowledge_retrieval" | "multi_system";
    complexity_score: number;
    stage: string;
    recommended_teams: Record<string, {
        reasoning: string;
        agents: string[];
        confidence?: number | undefined;
    }>;
    metadata?: Record<string, any> | undefined;
    estimated_cost?: number | undefined;
}, {
    workflow_id: string;
    timestamp: string;
    workflow_type: "sovereign" | "evolution" | "iterative_refinement" | "agent_collaboration" | "formal_verification" | "knowledge_retrieval" | "multi_system";
    complexity_score: number;
    stage: string;
    recommended_teams: Record<string, {
        reasoning: string;
        agents: string[];
        confidence?: number | undefined;
    }>;
    metadata?: Record<string, any> | undefined;
    estimated_cost?: number | undefined;
}>;
export type TeamSelectionResult = z.infer<typeof TeamSelectionResult>;
/**
 * Resource Optimization Result Schema (V2.0)
 */
export declare const ResourceOptimizationResult: z.ZodObject<{
    workflow_id: z.ZodString;
    stage: z.ZodString;
    complexity_score: z.ZodNumber;
    cpu_allocation: z.ZodNumber;
    memory_allocation_mb: z.ZodNumber;
    timeout_ms: z.ZodNumber;
    estimated_duration_ms: z.ZodOptional<z.ZodNumber>;
    estimated_cost_savings: z.ZodOptional<z.ZodNumber>;
    recommendations: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    workflow_id: string;
    timestamp: string;
    timeout_ms: number;
    complexity_score: number;
    stage: string;
    cpu_allocation: number;
    memory_allocation_mb: number;
    metadata?: Record<string, any> | undefined;
    recommendations?: string[] | undefined;
    estimated_duration_ms?: number | undefined;
    estimated_cost_savings?: number | undefined;
}, {
    workflow_id: string;
    timestamp: string;
    timeout_ms: number;
    complexity_score: number;
    stage: string;
    cpu_allocation: number;
    memory_allocation_mb: number;
    metadata?: Record<string, any> | undefined;
    recommendations?: string[] | undefined;
    estimated_duration_ms?: number | undefined;
    estimated_cost_savings?: number | undefined;
}>;
export type ResourceOptimizationResult = z.infer<typeof ResourceOptimizationResult>;
/**
 * Gauntlet Type Enum (V2.0)
 */
export declare const GauntletType: z.ZodEnum<["ADVERSARIAL", "FORMAL_VERIFICATION", "RED_TEAM", "CHAOS_ENGINEERING", "SECURITY_AUDIT", "PERFORMANCE_TEST", "CORRECTNESS_PROOF", "STRESS_TEST", "FUZZING", "COMPLIANCE_CHECK"]>;
export type GauntletType = z.infer<typeof GauntletType>;
/**
 * Gauntlet Severity Enum
 */
export declare const GauntletSeverity: z.ZodEnum<["BASIC", "STANDARD", "STRICT", "HARDCORE"]>;
export type GauntletSeverity = z.infer<typeof GauntletSeverity>;
/**
 * Gauntlet Config Schema (V2.0)
 */
export declare const GauntletConfig: z.ZodObject<{
    gauntlet_type: z.ZodEnum<["ADVERSARIAL", "FORMAL_VERIFICATION", "RED_TEAM", "CHAOS_ENGINEERING", "SECURITY_AUDIT", "PERFORMANCE_TEST", "CORRECTNESS_PROOF", "STRESS_TEST", "FUZZING", "COMPLIANCE_CHECK"]>;
    complexity_score: z.ZodNumber;
    severity: z.ZodOptional<z.ZodEnum<["BASIC", "STANDARD", "STRICT", "HARDCORE"]>>;
    custom_parameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timeout_ms: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
    complexity_score: number;
    timeout_ms?: number | undefined;
    severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
    custom_parameters?: Record<string, any> | undefined;
}, {
    gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
    complexity_score: number;
    timeout_ms?: number | undefined;
    severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
    custom_parameters?: Record<string, any> | undefined;
}>;
export type GauntletConfig = z.infer<typeof GauntletConfig>;
/**
 * Gauntlet Result Schema (V2.0)
 */
export declare const GauntletResult: z.ZodObject<{
    gauntlet_type: z.ZodEnum<["ADVERSARIAL", "FORMAL_VERIFICATION", "RED_TEAM", "CHAOS_ENGINEERING", "SECURITY_AUDIT", "PERFORMANCE_TEST", "CORRECTNESS_PROOF", "STRESS_TEST", "FUZZING", "COMPLIANCE_CHECK"]>;
    passed: z.ZodBoolean;
    score: z.ZodNumber;
    reasoning: z.ZodString;
    red_flags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    execution_time_ms: z.ZodNumber;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    score: number;
    passed: boolean;
    gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
    execution_time_ms: number;
    reasoning: string;
    red_flags?: string[] | undefined;
}, {
    timestamp: string;
    score: number;
    passed: boolean;
    gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
    execution_time_ms: number;
    reasoning: string;
    red_flags?: string[] | undefined;
}>;
export type GauntletResult = z.infer<typeof GauntletResult>;
/**
 * Gauntlet Pipeline Schema (V2.0)
 */
export declare const GauntletPipeline: z.ZodObject<{
    pipeline_id: z.ZodString;
    complexity_score: z.ZodNumber;
    base_gauntlet_type: z.ZodEnum<["ADVERSARIAL", "FORMAL_VERIFICATION", "RED_TEAM", "CHAOS_ENGINEERING", "SECURITY_AUDIT", "PERFORMANCE_TEST", "CORRECTNESS_PROOF", "STRESS_TEST", "FUZZING", "COMPLIANCE_CHECK"]>;
    gauntlets: z.ZodArray<z.ZodObject<{
        gauntlet_type: z.ZodEnum<["ADVERSARIAL", "FORMAL_VERIFICATION", "RED_TEAM", "CHAOS_ENGINEERING", "SECURITY_AUDIT", "PERFORMANCE_TEST", "CORRECTNESS_PROOF", "STRESS_TEST", "FUZZING", "COMPLIANCE_CHECK"]>;
        complexity_score: z.ZodNumber;
        severity: z.ZodOptional<z.ZodEnum<["BASIC", "STANDARD", "STRICT", "HARDCORE"]>>;
        custom_parameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        complexity_score: number;
        timeout_ms?: number | undefined;
        severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
        custom_parameters?: Record<string, any> | undefined;
    }, {
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        complexity_score: number;
        timeout_ms?: number | undefined;
        severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
        custom_parameters?: Record<string, any> | undefined;
    }>, "many">;
    execution_mode: z.ZodEnum<["sequential", "parallel", "adaptive"]>;
    aggregation_strategy: z.ZodEnum<["all_must_pass", "majority", "weighted"]>;
    severity: z.ZodOptional<z.ZodEnum<["BASIC", "STANDARD", "STRICT", "HARDCORE"]>>;
}, "strip", z.ZodTypeAny, {
    gauntlets: {
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        complexity_score: number;
        timeout_ms?: number | undefined;
        severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
        custom_parameters?: Record<string, any> | undefined;
    }[];
    complexity_score: number;
    pipeline_id: string;
    base_gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
    execution_mode: "adaptive" | "sequential" | "parallel";
    aggregation_strategy: "majority" | "all_must_pass" | "weighted";
    severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
}, {
    gauntlets: {
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        complexity_score: number;
        timeout_ms?: number | undefined;
        severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
        custom_parameters?: Record<string, any> | undefined;
    }[];
    complexity_score: number;
    pipeline_id: string;
    base_gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
    execution_mode: "adaptive" | "sequential" | "parallel";
    aggregation_strategy: "majority" | "all_must_pass" | "weighted";
    severity?: "BASIC" | "STANDARD" | "STRICT" | "HARDCORE" | undefined;
}>;
export type GauntletPipeline = z.infer<typeof GauntletPipeline>;
/**
 * Gauntlet Pipeline Result Schema (V2.0)
 */
export declare const GauntletPipelineResult: z.ZodObject<{
    pipeline_id: z.ZodString;
    total_gauntlets: z.ZodNumber;
    passed_gauntlets: z.ZodNumber;
    failed_gauntlets: z.ZodNumber;
    skipped_gauntlets: z.ZodOptional<z.ZodNumber>;
    overall_pass: z.ZodBoolean;
    aggregate_score: z.ZodNumber;
    gauntlet_results: z.ZodArray<z.ZodObject<{
        gauntlet_type: z.ZodEnum<["ADVERSARIAL", "FORMAL_VERIFICATION", "RED_TEAM", "CHAOS_ENGINEERING", "SECURITY_AUDIT", "PERFORMANCE_TEST", "CORRECTNESS_PROOF", "STRESS_TEST", "FUZZING", "COMPLIANCE_CHECK"]>;
        passed: z.ZodBoolean;
        score: z.ZodNumber;
        reasoning: z.ZodString;
        red_flags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        execution_time_ms: z.ZodNumber;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        score: number;
        passed: boolean;
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        execution_time_ms: number;
        reasoning: string;
        red_flags?: string[] | undefined;
    }, {
        timestamp: string;
        score: number;
        passed: boolean;
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        execution_time_ms: number;
        reasoning: string;
        red_flags?: string[] | undefined;
    }>, "many">;
    execution_time_ms: z.ZodNumber;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    pipeline_id: string;
    execution_time_ms: number;
    gauntlet_results: {
        timestamp: string;
        score: number;
        passed: boolean;
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        execution_time_ms: number;
        reasoning: string;
        red_flags?: string[] | undefined;
    }[];
    total_gauntlets: number;
    passed_gauntlets: number;
    failed_gauntlets: number;
    overall_pass: boolean;
    aggregate_score: number;
    skipped_gauntlets?: number | undefined;
}, {
    timestamp: string;
    pipeline_id: string;
    execution_time_ms: number;
    gauntlet_results: {
        timestamp: string;
        score: number;
        passed: boolean;
        gauntlet_type: "FORMAL_VERIFICATION" | "ADVERSARIAL" | "RED_TEAM" | "CHAOS_ENGINEERING" | "SECURITY_AUDIT" | "PERFORMANCE_TEST" | "CORRECTNESS_PROOF" | "STRESS_TEST" | "FUZZING" | "COMPLIANCE_CHECK";
        execution_time_ms: number;
        reasoning: string;
        red_flags?: string[] | undefined;
    }[];
    total_gauntlets: number;
    passed_gauntlets: number;
    failed_gauntlets: number;
    overall_pass: boolean;
    aggregate_score: number;
    skipped_gauntlets?: number | undefined;
}>;
export type GauntletPipelineResult = z.infer<typeof GauntletPipelineResult>;
/**
 * ICR Pattern Type Enum (V2.0)
 */
export declare const ICRPatternType: z.ZodEnum<["WORKFLOW_EXECUTION", "REFINEMENT_LOOP", "RESOURCE_USAGE", "QUALITY_OUTCOME", "RETRY_PATTERN", "BOTTLENECK", "OPTIMIZATION", "SECURITY_POLICY", "GAUNTLET_OUTCOME"]>;
export type ICRPatternType = z.infer<typeof ICRPatternType>;
/**
 * ICR Pattern Schema (V2.0)
 */
export declare const ICRPattern: z.ZodObject<{
    pattern_id: z.ZodString;
    pattern_type: z.ZodEnum<["WORKFLOW_EXECUTION", "REFINEMENT_LOOP", "RESOURCE_USAGE", "QUALITY_OUTCOME", "RETRY_PATTERN", "BOTTLENECK", "OPTIMIZATION", "SECURITY_POLICY", "GAUNTLET_OUTCOME"]>;
    context: z.ZodRecord<z.ZodString, z.ZodAny>;
    passed: z.ZodBoolean;
    metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    passed: boolean;
    context: Record<string, any>;
    pattern_id: string;
    pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
    metadata?: Record<string, any> | undefined;
    metrics?: Record<string, any> | undefined;
}, {
    timestamp: string;
    passed: boolean;
    context: Record<string, any>;
    pattern_id: string;
    pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
    metadata?: Record<string, any> | undefined;
    metrics?: Record<string, any> | undefined;
}>;
export type ICRPattern = z.infer<typeof ICRPattern>;
/**
 * ICR Prediction Schema (V2.0)
 */
export declare const ICRPrediction: z.ZodObject<{
    pattern_type: z.ZodEnum<["WORKFLOW_EXECUTION", "REFINEMENT_LOOP", "RESOURCE_USAGE", "QUALITY_OUTCOME", "RETRY_PATTERN", "BOTTLENECK", "OPTIMIZATION", "SECURITY_POLICY", "GAUNTLET_OUTCOME"]>;
    predicted_outcome: z.ZodBoolean;
    confidence: z.ZodNumber;
    recommended_action: z.ZodString;
    pattern_count: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    confidence: number;
    pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
    predicted_outcome: boolean;
    recommended_action: string;
    pattern_count?: number | undefined;
}, {
    timestamp: string;
    confidence: number;
    pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
    predicted_outcome: boolean;
    recommended_action: string;
    pattern_count?: number | undefined;
}>;
export type ICRPrediction = z.infer<typeof ICRPrediction>;
/**
 * Pattern Cluster Schema (V2.0)
 */
export declare const PatternCluster: z.ZodObject<{
    cluster_id: z.ZodString;
    pattern_type: z.ZodEnum<["WORKFLOW_EXECUTION", "REFINEMENT_LOOP", "RESOURCE_USAGE", "QUALITY_OUTCOME", "RETRY_PATTERN", "BOTTLENECK", "OPTIMIZATION", "SECURITY_POLICY", "GAUNTLET_OUTCOME"]>;
    patterns: z.ZodArray<z.ZodLazy<z.ZodObject<{
        pattern_id: z.ZodString;
        pattern_type: z.ZodEnum<["WORKFLOW_EXECUTION", "REFINEMENT_LOOP", "RESOURCE_USAGE", "QUALITY_OUTCOME", "RETRY_PATTERN", "BOTTLENECK", "OPTIMIZATION", "SECURITY_POLICY", "GAUNTLET_OUTCOME"]>;
        context: z.ZodRecord<z.ZodString, z.ZodAny>;
        passed: z.ZodBoolean;
        metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        timestamp: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        passed: boolean;
        context: Record<string, any>;
        pattern_id: string;
        pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
        metadata?: Record<string, any> | undefined;
        metrics?: Record<string, any> | undefined;
    }, {
        timestamp: string;
        passed: boolean;
        context: Record<string, any>;
        pattern_id: string;
        pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
        metadata?: Record<string, any> | undefined;
        metrics?: Record<string, any> | undefined;
    }>>, "many">;
    centroid: z.ZodRecord<z.ZodString, z.ZodAny>;
    similarity_score: z.ZodNumber;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
    similarity_score: number;
    patterns: {
        timestamp: string;
        passed: boolean;
        context: Record<string, any>;
        pattern_id: string;
        pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
        metadata?: Record<string, any> | undefined;
        metrics?: Record<string, any> | undefined;
    }[];
    cluster_id: string;
    centroid: Record<string, any>;
}, {
    timestamp: string;
    pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
    similarity_score: number;
    patterns: {
        timestamp: string;
        passed: boolean;
        context: Record<string, any>;
        pattern_id: string;
        pattern_type: "WORKFLOW_EXECUTION" | "REFINEMENT_LOOP" | "RESOURCE_USAGE" | "QUALITY_OUTCOME" | "RETRY_PATTERN" | "BOTTLENECK" | "OPTIMIZATION" | "SECURITY_POLICY" | "GAUNTLET_OUTCOME";
        metadata?: Record<string, any> | undefined;
        metrics?: Record<string, any> | undefined;
    }[];
    cluster_id: string;
    centroid: Record<string, any>;
}>;
export type PatternCluster = z.infer<typeof PatternCluster>;
/**
 * ICR Pattern Insights Schema (V2.0)
 */
export declare const ICRPatternInsights: z.ZodObject<{
    available: z.ZodBoolean;
    pattern_types: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
        count: z.ZodNumber;
        pass_rate: z.ZodNumber;
        confidence: z.ZodNumber;
        recent_patterns: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
    }, "strip", z.ZodTypeAny, {
        count: number;
        confidence: number;
        pass_rate: number;
        recent_patterns?: any[] | undefined;
    }, {
        count: number;
        confidence: number;
        pass_rate: number;
        recent_patterns?: any[] | undefined;
    }>>>;
    total_patterns: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    available: boolean;
    pattern_types?: Record<string, {
        count: number;
        confidence: number;
        pass_rate: number;
        recent_patterns?: any[] | undefined;
    }> | undefined;
    total_patterns?: number | undefined;
}, {
    timestamp: string;
    available: boolean;
    pattern_types?: Record<string, {
        count: number;
        confidence: number;
        pass_rate: number;
        recent_patterns?: any[] | undefined;
    }> | undefined;
    total_patterns?: number | undefined;
}>;
export type ICRPatternInsights = z.infer<typeof ICRPatternInsights>;
/**
 * Chart Type Enum (V2.0 - UI)
 */
export declare const ChartType: z.ZodEnum<["RADAR", "BAR", "LINE", "PIE", "TIMELINE", "SCATTER", "HEATMAP"]>;
export type ChartType = z.infer<typeof ChartType>;
/**
 * UI Chart Data Schema (V2.0)
 */
export declare const UIChartData: z.ZodObject<{
    chart_type: z.ZodEnum<["RADAR", "BAR", "LINE", "PIE", "TIMELINE", "SCATTER", "HEATMAP"]>;
    title: z.ZodString;
    labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    datasets: z.ZodOptional<z.ZodArray<z.ZodObject<{
        label: z.ZodOptional<z.ZodString>;
        data: z.ZodArray<z.ZodUnion<[z.ZodNumber, z.ZodString, z.ZodBoolean, z.ZodArray<z.ZodAny, "many">]>, "many">;
        backgroundColor: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        borderColor: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        data: (string | number | boolean | any[])[];
        label?: string | undefined;
        backgroundColor?: string[] | undefined;
        borderColor?: string[] | undefined;
    }, {
        data: (string | number | boolean | any[])[];
        label?: string | undefined;
        backgroundColor?: string[] | undefined;
        borderColor?: string[] | undefined;
    }>, "many">>;
    data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    recommendations: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    title: string;
    chart_type: "RADAR" | "BAR" | "LINE" | "PIE" | "TIMELINE" | "SCATTER" | "HEATMAP";
    data?: Record<string, any> | undefined;
    recommendations?: string[] | undefined;
    options?: Record<string, any> | undefined;
    labels?: string[] | undefined;
    datasets?: {
        data: (string | number | boolean | any[])[];
        label?: string | undefined;
        backgroundColor?: string[] | undefined;
        borderColor?: string[] | undefined;
    }[] | undefined;
}, {
    timestamp: string;
    title: string;
    chart_type: "RADAR" | "BAR" | "LINE" | "PIE" | "TIMELINE" | "SCATTER" | "HEATMAP";
    data?: Record<string, any> | undefined;
    recommendations?: string[] | undefined;
    options?: Record<string, any> | undefined;
    labels?: string[] | undefined;
    datasets?: {
        data: (string | number | boolean | any[])[];
        label?: string | undefined;
        backgroundColor?: string[] | undefined;
        borderColor?: string[] | undefined;
    }[] | undefined;
}>;
export type UIChartData = z.infer<typeof UIChartData>;
/**
 * Workflow Timeline Schema (V2.0 - UI)
 */
export declare const WorkflowTimeline: z.ZodObject<{
    chart_type: z.ZodEnum<["RADAR", "BAR", "LINE", "PIE", "TIMELINE", "SCATTER", "HEATMAP"]>;
    workflow_id: z.ZodString;
    stages: z.ZodArray<z.ZodObject<{
        stage: z.ZodString;
        status: z.ZodEnum<["pending", "in_progress", "completed", "failed", "skipped"]>;
        duration_ms: z.ZodOptional<z.ZodNumber>;
        start_time: z.ZodOptional<z.ZodString>;
        end_time: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        status: "completed" | "failed" | "pending" | "in_progress" | "skipped";
        stage: string;
        duration_ms?: number | undefined;
        start_time?: string | undefined;
        end_time?: string | undefined;
    }, {
        status: "completed" | "failed" | "pending" | "in_progress" | "skipped";
        stage: string;
        duration_ms?: number | undefined;
        start_time?: string | undefined;
        end_time?: string | undefined;
    }>, "many">;
    total_duration_ms: z.ZodNumber;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    workflow_id: string;
    timestamp: string;
    total_duration_ms: number;
    chart_type: "RADAR" | "BAR" | "LINE" | "PIE" | "TIMELINE" | "SCATTER" | "HEATMAP";
    stages: {
        status: "completed" | "failed" | "pending" | "in_progress" | "skipped";
        stage: string;
        duration_ms?: number | undefined;
        start_time?: string | undefined;
        end_time?: string | undefined;
    }[];
}, {
    workflow_id: string;
    timestamp: string;
    total_duration_ms: number;
    chart_type: "RADAR" | "BAR" | "LINE" | "PIE" | "TIMELINE" | "SCATTER" | "HEATMAP";
    stages: {
        status: "completed" | "failed" | "pending" | "in_progress" | "skipped";
        stage: string;
        duration_ms?: number | undefined;
        start_time?: string | undefined;
        end_time?: string | undefined;
    }[];
}>;
export type WorkflowTimeline = z.infer<typeof WorkflowTimeline>;
/**
 * Adapter Health Status Schema (V2.0)
 */
export declare const AdapterHealthStatus: z.ZodObject<{
    overall_status: z.ZodEnum<["healthy", "degraded", "unhealthy"]>;
    components: z.ZodRecord<z.ZodString, z.ZodObject<{
        status: z.ZodEnum<["healthy", "degraded", "unhealthy", "disabled"]>;
        last_check: z.ZodOptional<z.ZodString>;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        error?: string | undefined;
        last_check?: string | undefined;
    }, {
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        error?: string | undefined;
        last_check?: string | undefined;
    }>>;
    uptime_ms: z.ZodOptional<z.ZodNumber>;
    alerts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        severity: z.ZodEnum<["info", "warning", "error", "critical"]>;
        message: z.ZodString;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        message: string;
        severity: "info" | "error" | "warning" | "critical";
    }, {
        timestamp: string;
        message: string;
        severity: "info" | "error" | "warning" | "critical";
    }>, "many">>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    components: Record<string, {
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        error?: string | undefined;
        last_check?: string | undefined;
    }>;
    overall_status: "healthy" | "unhealthy" | "degraded";
    alerts?: {
        timestamp: string;
        message: string;
        severity: "info" | "error" | "warning" | "critical";
    }[] | undefined;
    uptime_ms?: number | undefined;
}, {
    timestamp: string;
    components: Record<string, {
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        error?: string | undefined;
        last_check?: string | undefined;
    }>;
    overall_status: "healthy" | "unhealthy" | "degraded";
    alerts?: {
        timestamp: string;
        message: string;
        severity: "info" | "error" | "warning" | "critical";
    }[] | undefined;
    uptime_ms?: number | undefined;
}>;
export type AdapterHealthStatus = z.infer<typeof AdapterHealthStatus>;
/**
 * Cache Statistics Schema (V2.0 - Performance)
 */
export declare const CacheStatistics: z.ZodObject<{
    size: z.ZodNumber;
    max_size: z.ZodNumber;
    ttl: z.ZodNumber;
    total_hits: z.ZodNumber;
    total_misses: z.ZodNumber;
    hit_rate: z.ZodNumber;
    evictions: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    size: number;
    max_size: number;
    ttl: number;
    total_hits: number;
    total_misses: number;
    hit_rate: number;
    evictions?: number | undefined;
}, {
    timestamp: string;
    size: number;
    max_size: number;
    ttl: number;
    total_hits: number;
    total_misses: number;
    hit_rate: number;
    evictions?: number | undefined;
}>;
export type CacheStatistics = z.infer<typeof CacheStatistics>;
/**
 * Performance Metrics Schema (V2.0)
 */
export declare const PerformanceMetrics: z.ZodObject<{
    operation: z.ZodString;
    count: z.ZodNumber;
    avg_ms: z.ZodNumber;
    min_ms: z.ZodNumber;
    max_ms: z.ZodNumber;
    p50_ms: z.ZodNumber;
    p95_ms: z.ZodNumber;
    p99_ms: z.ZodNumber;
    throughput_per_sec: z.ZodOptional<z.ZodNumber>;
    error_rate: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    count: number;
    operation: string;
    avg_ms: number;
    min_ms: number;
    max_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    error_rate?: number | undefined;
    throughput_per_sec?: number | undefined;
}, {
    timestamp: string;
    count: number;
    operation: string;
    avg_ms: number;
    min_ms: number;
    max_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    error_rate?: number | undefined;
    throughput_per_sec?: number | undefined;
}>;
export type PerformanceMetrics = z.infer<typeof PerformanceMetrics>;
/**
 * Async Operation Status Enum (V2.0)
 */
export declare const AsyncOperationStatus: z.ZodEnum<["pending", "scheduled", "running", "completed", "failed", "cancelled"]>;
export type AsyncOperationStatus = z.infer<typeof AsyncOperationStatus>;
/**
 * Async Operation Schema (V2.0)
 */
export declare const AsyncOperation: z.ZodObject<{
    operation_id: z.ZodString;
    operation_type: z.ZodString;
    status: z.ZodEnum<["pending", "scheduled", "running", "completed", "failed", "cancelled"]>;
    input_data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    result: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    created_at: z.ZodString;
    started_at: z.ZodOptional<z.ZodString>;
    completed_at: z.ZodOptional<z.ZodString>;
    duration_ms: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
    created_at: string;
    operation_id: string;
    operation_type: string;
    error?: string | undefined;
    metadata?: Record<string, any> | undefined;
    duration_ms?: number | undefined;
    result?: any;
    input_data?: Record<string, any> | undefined;
    started_at?: string | undefined;
    completed_at?: string | undefined;
}, {
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
    created_at: string;
    operation_id: string;
    operation_type: string;
    error?: string | undefined;
    metadata?: Record<string, any> | undefined;
    duration_ms?: number | undefined;
    result?: any;
    input_data?: Record<string, any> | undefined;
    started_at?: string | undefined;
    completed_at?: string | undefined;
}>;
export type AsyncOperation = z.infer<typeof AsyncOperation>;
/**
 * Batch Operation Schema (V2.0)
 */
export declare const BatchOperation: z.ZodObject<{
    batch_id: z.ZodString;
    operations: z.ZodArray<z.ZodObject<{
        operation_id: z.ZodString;
        operation_type: z.ZodString;
        status: z.ZodEnum<["pending", "scheduled", "running", "completed", "failed", "cancelled"]>;
        input_data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        result: z.ZodOptional<z.ZodAny>;
        error: z.ZodOptional<z.ZodString>;
        created_at: z.ZodString;
        started_at: z.ZodOptional<z.ZodString>;
        completed_at: z.ZodOptional<z.ZodString>;
        duration_ms: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
        created_at: string;
        operation_id: string;
        operation_type: string;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        duration_ms?: number | undefined;
        result?: any;
        input_data?: Record<string, any> | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
    }, {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
        created_at: string;
        operation_id: string;
        operation_type: string;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        duration_ms?: number | undefined;
        result?: any;
        input_data?: Record<string, any> | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
    }>, "many">;
    total_operations: z.ZodNumber;
    completed_operations: z.ZodNumber;
    failed_operations: z.ZodNumber;
    max_concurrency: z.ZodOptional<z.ZodNumber>;
    status: z.ZodEnum<["pending", "scheduled", "running", "completed", "failed", "cancelled"]>;
    created_at: z.ZodString;
    completed_at: z.ZodOptional<z.ZodString>;
    total_duration_ms: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
    created_at: string;
    batch_id: string;
    operations: {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
        created_at: string;
        operation_id: string;
        operation_type: string;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        duration_ms?: number | undefined;
        result?: any;
        input_data?: Record<string, any> | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
    }[];
    total_operations: number;
    completed_operations: number;
    failed_operations: number;
    completed_at?: string | undefined;
    total_duration_ms?: number | undefined;
    max_concurrency?: number | undefined;
}, {
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
    created_at: string;
    batch_id: string;
    operations: {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "scheduled";
        created_at: string;
        operation_id: string;
        operation_type: string;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        duration_ms?: number | undefined;
        result?: any;
        input_data?: Record<string, any> | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
    }[];
    total_operations: number;
    completed_operations: number;
    failed_operations: number;
    completed_at?: string | undefined;
    total_duration_ms?: number | undefined;
    max_concurrency?: number | undefined;
}>;
export type BatchOperation = z.infer<typeof BatchOperation>;
/**
 * Additional System Type Enum (V2.0)
 */
export declare const AdditionalSystemType: z.ZodEnum<["crewai", "mcp_tools", "knowledge_engine", "leanaide", "z3_prover"]>;
export type AdditionalSystemType = z.infer<typeof AdditionalSystemType>;
/**
 * System Health Schema (V2.0)
 */
export declare const SystemHealth: z.ZodObject<{
    system: z.ZodEnum<["crewai", "mcp_tools", "knowledge_engine", "leanaide", "z3_prover"]>;
    available: z.ZodBoolean;
    status: z.ZodEnum<["healthy", "degraded", "unhealthy", "disabled"]>;
    last_check: z.ZodOptional<z.ZodString>;
    reason: z.ZodOptional<z.ZodString>;
    capabilities: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "healthy" | "unhealthy" | "disabled" | "degraded";
    system: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover";
    available: boolean;
    metadata?: Record<string, any> | undefined;
    capabilities?: string[] | undefined;
    reason?: string | undefined;
    last_check?: string | undefined;
}, {
    timestamp: string;
    status: "healthy" | "unhealthy" | "disabled" | "degraded";
    system: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover";
    available: boolean;
    metadata?: Record<string, any> | undefined;
    capabilities?: string[] | undefined;
    reason?: string | undefined;
    last_check?: string | undefined;
}>;
export type SystemHealth = z.infer<typeof SystemHealth>;
/**
 * Unified System Health Schema (V2.0)
 */
export declare const UnifiedSystemHealth: z.ZodObject<{
    overall_status: z.ZodEnum<["healthy", "degraded", "unhealthy"]>;
    total_systems: z.ZodNumber;
    available_systems: z.ZodNumber;
    systems: z.ZodRecord<z.ZodString, z.ZodObject<{
        system: z.ZodEnum<["crewai", "mcp_tools", "knowledge_engine", "leanaide", "z3_prover"]>;
        available: z.ZodBoolean;
        status: z.ZodEnum<["healthy", "degraded", "unhealthy", "disabled"]>;
        last_check: z.ZodOptional<z.ZodString>;
        reason: z.ZodOptional<z.ZodString>;
        capabilities: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        system: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover";
        available: boolean;
        metadata?: Record<string, any> | undefined;
        capabilities?: string[] | undefined;
        reason?: string | undefined;
        last_check?: string | undefined;
    }, {
        timestamp: string;
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        system: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover";
        available: boolean;
        metadata?: Record<string, any> | undefined;
        capabilities?: string[] | undefined;
        reason?: string | undefined;
        last_check?: string | undefined;
    }>>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    overall_status: "healthy" | "unhealthy" | "degraded";
    systems: Record<string, {
        timestamp: string;
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        system: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover";
        available: boolean;
        metadata?: Record<string, any> | undefined;
        capabilities?: string[] | undefined;
        reason?: string | undefined;
        last_check?: string | undefined;
    }>;
    total_systems: number;
    available_systems: number;
}, {
    timestamp: string;
    overall_status: "healthy" | "unhealthy" | "degraded";
    systems: Record<string, {
        timestamp: string;
        status: "healthy" | "unhealthy" | "disabled" | "degraded";
        system: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover";
        available: boolean;
        metadata?: Record<string, any> | undefined;
        capabilities?: string[] | undefined;
        reason?: string | undefined;
        last_check?: string | undefined;
    }>;
    total_systems: number;
    available_systems: number;
}>;
export type UnifiedSystemHealth = z.infer<typeof UnifiedSystemHealth>;
/**
 * Workflow Step Schema (V2.0)
 */
export declare const WorkflowStep: z.ZodObject<{
    step: z.ZodNumber;
    system: z.ZodOptional<z.ZodEnum<["crewai", "mcp_tools", "knowledge_engine", "leanaide", "z3_prover"]>>;
    action: z.ZodString;
    success: z.ZodBoolean;
    result: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    duration_ms: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    action: string;
    step: number;
    error?: string | undefined;
    duration_ms?: number | undefined;
    result?: any;
    system?: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover" | undefined;
}, {
    timestamp: string;
    success: boolean;
    action: string;
    step: number;
    error?: string | undefined;
    duration_ms?: number | undefined;
    result?: any;
    system?: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover" | undefined;
}>;
export type WorkflowStep = z.infer<typeof WorkflowStep>;
/**
 * Cross-System Workflow Result Schema (V2.0)
 */
export declare const CrossSystemWorkflowResult: z.ZodObject<{
    workflow_type: z.ZodString;
    success: z.ZodBoolean;
    steps: z.ZodArray<z.ZodObject<{
        step: z.ZodNumber;
        system: z.ZodOptional<z.ZodEnum<["crewai", "mcp_tools", "knowledge_engine", "leanaide", "z3_prover"]>>;
        action: z.ZodString;
        success: z.ZodBoolean;
        result: z.ZodOptional<z.ZodAny>;
        error: z.ZodOptional<z.ZodString>;
        duration_ms: z.ZodOptional<z.ZodNumber>;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        success: boolean;
        action: string;
        step: number;
        error?: string | undefined;
        duration_ms?: number | undefined;
        result?: any;
        system?: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover" | undefined;
    }, {
        timestamp: string;
        success: boolean;
        action: string;
        step: number;
        error?: string | undefined;
        duration_ms?: number | undefined;
        result?: any;
        system?: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover" | undefined;
    }>, "many">;
    result_count: z.ZodOptional<z.ZodNumber>;
    result: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    total_duration_ms: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    steps: {
        timestamp: string;
        success: boolean;
        action: string;
        step: number;
        error?: string | undefined;
        duration_ms?: number | undefined;
        result?: any;
        system?: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover" | undefined;
    }[];
    workflow_type: string;
    error?: string | undefined;
    metadata?: Record<string, any> | undefined;
    result?: any;
    result_count?: number | undefined;
    total_duration_ms?: number | undefined;
}, {
    timestamp: string;
    success: boolean;
    steps: {
        timestamp: string;
        success: boolean;
        action: string;
        step: number;
        error?: string | undefined;
        duration_ms?: number | undefined;
        result?: any;
        system?: "leanaide" | "knowledge_engine" | "crewai" | "mcp_tools" | "z3_prover" | undefined;
    }[];
    workflow_type: string;
    error?: string | undefined;
    metadata?: Record<string, any> | undefined;
    result?: any;
    result_count?: number | undefined;
    total_duration_ms?: number | undefined;
}>;
export type CrossSystemWorkflowResult = z.infer<typeof CrossSystemWorkflowResult>;
/**
 * V2.0 Validation Functions
 */
export declare function validateProblemDecomposition(data: unknown): {
    success: boolean;
    data?: ProblemDecompositionResult;
    errors?: string[];
};
export declare function validateTeamSelection(data: unknown): {
    success: boolean;
    data?: TeamSelectionResult;
    errors?: string[];
};
export declare function validateGauntletPipelineResult(data: unknown): {
    success: boolean;
    data?: GauntletPipelineResult;
    errors?: string[];
};
export declare function validateICRPattern(data: unknown): {
    success: boolean;
    data?: ICRPattern;
    errors?: string[];
};
export declare function validateCrossSystemWorkflowResult(data: unknown): {
    success: boolean;
    data?: CrossSystemWorkflowResult;
    errors?: string[];
};
//# sourceMappingURL=adaptive-mdap-canonical.d.ts.map