/**
 * LoongFlow Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for LoongFlow interactions.
 * LoongFlow implements the Plan-Execute-Summarize (PES) paradigm with evolutionary
 * optimization using MAP-Elites and island models.
 *
 * Key LoongFlow Concepts:
 * - PES Paradigm: Plan → Execute → Summarize
 * - MAP-Elites: Quality-diversity optimization for maintaining diverse solution archive
 * - Island Model: Parallel evolution across multiple populations
 * - Boltzmann Sampling: Temperature-based selection for exploration/exploitation
 * - Workers: Planner, Executor, Summarizer agents
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for LoongFlow data in
 * the glue layer. Do not pass raw LoongFlow API responses between services.
 *
 * Reference: Task #1 scan results showing LoongFlow's Solution dataclass and
 * async worker interface.
 */
import { z } from 'zod';
import { type Problem as ProblemType, type ExecutionResult as ExecutionResultType } from './pes-canonical';
/**
 * LoongFlow State Enum
 *
 * Tracks the state of a LoongFlow execution.
 */
export declare const LoongFlowState: z.ZodEnum<["idle", "planning", "executing", "summarizing", "evolving", "completed", "failed", "cancelled"]>;
export type LoongFlowState = z.infer<typeof LoongFlowState>;
/**
 * LoongFlow Worker Type Enum
 *
 * Types of workers in the LoongFlow system.
 */
export declare const LoongFlowWorkerType: z.ZodEnum<["planner", "executor", "summarizer", "evolver"]>;
export type LoongFlowWorkerType = z.infer<typeof LoongFlowWorkerType>;
/**
 * ============================================================================
 * LOONGFLOW-SPECIFIC SCHEMAS
 * ============================================================================
 */
/**
 * LoongFlow Solution Schema
 *
 * Direct mapping to LoongFlow's Solution dataclass from core-projects.
 * This represents a single solution in the evolutionary population.
 *
 * From LoongFlow source (src/loongflow/agentsdk/memory/evolution/base_memory.py):
 * ```python
 * @dataclass
 * class Solution:
 *     # Solution identification
 *     solution: str = ""
 *     solution_id: str = ""
 *
 *     # Evolution information
 *     generate_plan: str = ""
 *     parent_id: Optional[str] = ""
 *     island_id: Optional[int] = 0
 *     iteration: Optional[int] = 0
 *     timestamp: float = field(default_factory=time.time)
 *     generation: int = 0
 *     sample_cnt: int = 0
 *     sample_weight: float = 0.0
 *
 *     # Performance metrics
 *     score: Optional[float] = 0.0
 *     evaluation: Optional[str] = ""
 *     summary: str = ""
 *
 *     # Metadata
 *     metadata: Dict[str, Any] = field(default_factory=dict)
 * ```
 */
export declare const LoongFlowSolution: z.ZodObject<{
    solution: z.ZodString;
    solution_id: z.ZodString;
    generate_plan: z.ZodString;
    parent_id: z.ZodString;
    island_id: z.ZodNumber;
    iteration: z.ZodNumber;
    timestamp: z.ZodNumber;
    generation: z.ZodNumber;
    sample_cnt: z.ZodNumber;
    sample_weight: z.ZodNumber;
    score: z.ZodNumber;
    evaluation: z.ZodString;
    summary: z.ZodString;
    metadata: z.ZodRecord<z.ZodString, z.ZodAny>;
}, "strip", z.ZodTypeAny, {
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    score: number;
    evaluation: string;
    iteration: number;
    island_id: number;
    solution_id: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
}, {
    timestamp: number;
    metadata: Record<string, any>;
    summary: string;
    solution: string;
    score: number;
    evaluation: string;
    iteration: number;
    island_id: number;
    solution_id: string;
    parent_id: string;
    generate_plan: string;
    generation: number;
    sample_cnt: number;
    sample_weight: number;
}>;
export type LoongFlowSolution = z.infer<typeof LoongFlowSolution>;
/**
 * LLM Configuration Schema
 *
 * Configuration for LLM providers used by LoongFlow.
 * Direct mapping to Python LLMConfig from src/loongflow/framework/pes/context/config.py
 */
export declare const LLMConfig: z.ZodObject<{
    model: z.ZodString;
    url: z.ZodOptional<z.ZodString>;
    api_key: z.ZodOptional<z.ZodString>;
    model_provider: z.ZodOptional<z.ZodString>;
    temperature: z.ZodOptional<z.ZodNumber>;
    context_length: z.ZodDefault<z.ZodNumber>;
    max_tokens: z.ZodDefault<z.ZodNumber>;
    top_p: z.ZodOptional<z.ZodNumber>;
    timeout: z.ZodDefault<z.ZodNumber>;
    completion_token_price: z.ZodDefault<z.ZodNumber>;
    prompt_token_price: z.ZodDefault<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    max_tokens: number;
    model: string;
    context_length: number;
    completion_token_price: number;
    prompt_token_price: number;
    url?: string | undefined;
    api_key?: string | undefined;
    temperature?: number | undefined;
    top_p?: number | undefined;
    model_provider?: string | undefined;
}, {
    model: string;
    url?: string | undefined;
    timeout?: number | undefined;
    api_key?: string | undefined;
    temperature?: number | undefined;
    top_p?: number | undefined;
    max_tokens?: number | undefined;
    model_provider?: string | undefined;
    context_length?: number | undefined;
    completion_token_price?: number | undefined;
    prompt_token_price?: number | undefined;
}>;
export type LLMConfig = z.infer<typeof LLMConfig>;
/**
 * Worker Configuration Schema
 *
 * Configuration for LoongFlow workers.
 */
export declare const WorkerConfig: z.ZodObject<{
    planner: z.ZodString;
    executor: z.ZodString;
    summarizer: z.ZodString;
    planner_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    executor_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    summarizer_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    executor: string;
    planner: string;
    summarizer: string;
    planner_config?: Record<string, any> | undefined;
    executor_config?: Record<string, any> | undefined;
    summarizer_config?: Record<string, any> | undefined;
}, {
    executor: string;
    planner: string;
    summarizer: string;
    planner_config?: Record<string, any> | undefined;
    executor_config?: Record<string, any> | undefined;
    summarizer_config?: Record<string, any> | undefined;
}>;
export type WorkerConfig = z.infer<typeof WorkerConfig>;
/**
 * Evolution Configuration Schema
 *
 * Configuration for evolutionary optimization in LoongFlow.
 */
export declare const EvolutionConfig: z.ZodObject<{
    iterations: z.ZodNumber;
    islands: z.ZodNumber;
    sample_size: z.ZodNumber;
    map_elites_enabled: z.ZodOptional<z.ZodBoolean>;
    map_dimensions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        bins: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        name: string;
        bins: number;
    }, {
        name: string;
        bins: number;
    }>, "many">>;
    temperature_initial: z.ZodOptional<z.ZodNumber>;
    temperature_decay: z.ZodOptional<z.ZodNumber>;
    migration_interval: z.ZodOptional<z.ZodNumber>;
    migration_size: z.ZodOptional<z.ZodNumber>;
    tournament_size: z.ZodOptional<z.ZodNumber>;
    elite_size: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    iterations: number;
    islands: number;
    sample_size: number;
    migration_interval?: number | undefined;
    map_elites_enabled?: boolean | undefined;
    map_dimensions?: {
        name: string;
        bins: number;
    }[] | undefined;
    temperature_initial?: number | undefined;
    temperature_decay?: number | undefined;
    migration_size?: number | undefined;
    tournament_size?: number | undefined;
    elite_size?: number | undefined;
}, {
    iterations: number;
    islands: number;
    sample_size: number;
    migration_interval?: number | undefined;
    map_elites_enabled?: boolean | undefined;
    map_dimensions?: {
        name: string;
        bins: number;
    }[] | undefined;
    temperature_initial?: number | undefined;
    temperature_decay?: number | undefined;
    migration_size?: number | undefined;
    tournament_size?: number | undefined;
    elite_size?: number | undefined;
}>;
export type EvolutionConfig = z.infer<typeof EvolutionConfig>;
/**
 * LoongFlow Configuration Schema
 *
 * Complete configuration for a LoongFlow execution.
 */
export declare const LoongFlowConfig: z.ZodObject<{
    task_name: z.ZodString;
    llm_config: z.ZodObject<{
        model: z.ZodString;
        url: z.ZodOptional<z.ZodString>;
        api_key: z.ZodOptional<z.ZodString>;
        model_provider: z.ZodOptional<z.ZodString>;
        temperature: z.ZodOptional<z.ZodNumber>;
        context_length: z.ZodDefault<z.ZodNumber>;
        max_tokens: z.ZodDefault<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        timeout: z.ZodDefault<z.ZodNumber>;
        completion_token_price: z.ZodDefault<z.ZodNumber>;
        prompt_token_price: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        max_tokens: number;
        model: string;
        context_length: number;
        completion_token_price: number;
        prompt_token_price: number;
        url?: string | undefined;
        api_key?: string | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_provider?: string | undefined;
    }, {
        model: string;
        url?: string | undefined;
        timeout?: number | undefined;
        api_key?: string | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        max_tokens?: number | undefined;
        model_provider?: string | undefined;
        context_length?: number | undefined;
        completion_token_price?: number | undefined;
        prompt_token_price?: number | undefined;
    }>;
    workers: z.ZodObject<{
        planner: z.ZodString;
        executor: z.ZodString;
        summarizer: z.ZodString;
        planner_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        executor_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        summarizer_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        executor: string;
        planner: string;
        summarizer: string;
        planner_config?: Record<string, any> | undefined;
        executor_config?: Record<string, any> | undefined;
        summarizer_config?: Record<string, any> | undefined;
    }, {
        executor: string;
        planner: string;
        summarizer: string;
        planner_config?: Record<string, any> | undefined;
        executor_config?: Record<string, any> | undefined;
        summarizer_config?: Record<string, any> | undefined;
    }>;
    evolution: z.ZodObject<{
        iterations: z.ZodNumber;
        islands: z.ZodNumber;
        sample_size: z.ZodNumber;
        map_elites_enabled: z.ZodOptional<z.ZodBoolean>;
        map_dimensions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            bins: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            name: string;
            bins: number;
        }, {
            name: string;
            bins: number;
        }>, "many">>;
        temperature_initial: z.ZodOptional<z.ZodNumber>;
        temperature_decay: z.ZodOptional<z.ZodNumber>;
        migration_interval: z.ZodOptional<z.ZodNumber>;
        migration_size: z.ZodOptional<z.ZodNumber>;
        tournament_size: z.ZodOptional<z.ZodNumber>;
        elite_size: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        iterations: number;
        islands: number;
        sample_size: number;
        migration_interval?: number | undefined;
        map_elites_enabled?: boolean | undefined;
        map_dimensions?: {
            name: string;
            bins: number;
        }[] | undefined;
        temperature_initial?: number | undefined;
        temperature_decay?: number | undefined;
        migration_size?: number | undefined;
        tournament_size?: number | undefined;
        elite_size?: number | undefined;
    }, {
        iterations: number;
        islands: number;
        sample_size: number;
        migration_interval?: number | undefined;
        map_elites_enabled?: boolean | undefined;
        map_dimensions?: {
            name: string;
            bins: number;
        }[] | undefined;
        temperature_initial?: number | undefined;
        temperature_decay?: number | undefined;
        migration_size?: number | undefined;
        tournament_size?: number | undefined;
        elite_size?: number | undefined;
    }>;
    timeout_ms: z.ZodOptional<z.ZodNumber>;
    checkpoint_interval: z.ZodOptional<z.ZodNumber>;
    checkpoint_dir: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    evolution: {
        iterations: number;
        islands: number;
        sample_size: number;
        migration_interval?: number | undefined;
        map_elites_enabled?: boolean | undefined;
        map_dimensions?: {
            name: string;
            bins: number;
        }[] | undefined;
        temperature_initial?: number | undefined;
        temperature_decay?: number | undefined;
        migration_size?: number | undefined;
        tournament_size?: number | undefined;
        elite_size?: number | undefined;
    };
    workers: {
        executor: string;
        planner: string;
        summarizer: string;
        planner_config?: Record<string, any> | undefined;
        executor_config?: Record<string, any> | undefined;
        summarizer_config?: Record<string, any> | undefined;
    };
    task_name: string;
    llm_config: {
        timeout: number;
        max_tokens: number;
        model: string;
        context_length: number;
        completion_token_price: number;
        prompt_token_price: number;
        url?: string | undefined;
        api_key?: string | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_provider?: string | undefined;
    };
    timeout_ms?: number | undefined;
    metadata?: Record<string, any> | undefined;
    checkpoint_interval?: number | undefined;
    checkpoint_dir?: string | undefined;
}, {
    evolution: {
        iterations: number;
        islands: number;
        sample_size: number;
        migration_interval?: number | undefined;
        map_elites_enabled?: boolean | undefined;
        map_dimensions?: {
            name: string;
            bins: number;
        }[] | undefined;
        temperature_initial?: number | undefined;
        temperature_decay?: number | undefined;
        migration_size?: number | undefined;
        tournament_size?: number | undefined;
        elite_size?: number | undefined;
    };
    workers: {
        executor: string;
        planner: string;
        summarizer: string;
        planner_config?: Record<string, any> | undefined;
        executor_config?: Record<string, any> | undefined;
        summarizer_config?: Record<string, any> | undefined;
    };
    task_name: string;
    llm_config: {
        model: string;
        url?: string | undefined;
        timeout?: number | undefined;
        api_key?: string | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        max_tokens?: number | undefined;
        model_provider?: string | undefined;
        context_length?: number | undefined;
        completion_token_price?: number | undefined;
        prompt_token_price?: number | undefined;
    };
    timeout_ms?: number | undefined;
    metadata?: Record<string, any> | undefined;
    checkpoint_interval?: number | undefined;
    checkpoint_dir?: string | undefined;
}>;
export type LoongFlowConfig = z.infer<typeof LoongFlowConfig>;
/**
 * LoongFlow Request Schema
 *
 * Represents a request to execute LoongFlow on a problem.
 */
export declare const LoongFlowRequest: z.ZodObject<{
    problem_id: z.ZodString;
    task_config: z.ZodObject<{
        task_name: z.ZodString;
        llm_config: z.ZodObject<{
            model: z.ZodString;
            url: z.ZodOptional<z.ZodString>;
            api_key: z.ZodOptional<z.ZodString>;
            model_provider: z.ZodOptional<z.ZodString>;
            temperature: z.ZodOptional<z.ZodNumber>;
            context_length: z.ZodDefault<z.ZodNumber>;
            max_tokens: z.ZodDefault<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            timeout: z.ZodDefault<z.ZodNumber>;
            completion_token_price: z.ZodDefault<z.ZodNumber>;
            prompt_token_price: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            timeout: number;
            max_tokens: number;
            model: string;
            context_length: number;
            completion_token_price: number;
            prompt_token_price: number;
            url?: string | undefined;
            api_key?: string | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_provider?: string | undefined;
        }, {
            model: string;
            url?: string | undefined;
            timeout?: number | undefined;
            api_key?: string | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            max_tokens?: number | undefined;
            model_provider?: string | undefined;
            context_length?: number | undefined;
            completion_token_price?: number | undefined;
            prompt_token_price?: number | undefined;
        }>;
        workers: z.ZodObject<{
            planner: z.ZodString;
            executor: z.ZodString;
            summarizer: z.ZodString;
            planner_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            executor_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            summarizer_config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            executor: string;
            planner: string;
            summarizer: string;
            planner_config?: Record<string, any> | undefined;
            executor_config?: Record<string, any> | undefined;
            summarizer_config?: Record<string, any> | undefined;
        }, {
            executor: string;
            planner: string;
            summarizer: string;
            planner_config?: Record<string, any> | undefined;
            executor_config?: Record<string, any> | undefined;
            summarizer_config?: Record<string, any> | undefined;
        }>;
        evolution: z.ZodObject<{
            iterations: z.ZodNumber;
            islands: z.ZodNumber;
            sample_size: z.ZodNumber;
            map_elites_enabled: z.ZodOptional<z.ZodBoolean>;
            map_dimensions: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                bins: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                name: string;
                bins: number;
            }, {
                name: string;
                bins: number;
            }>, "many">>;
            temperature_initial: z.ZodOptional<z.ZodNumber>;
            temperature_decay: z.ZodOptional<z.ZodNumber>;
            migration_interval: z.ZodOptional<z.ZodNumber>;
            migration_size: z.ZodOptional<z.ZodNumber>;
            tournament_size: z.ZodOptional<z.ZodNumber>;
            elite_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            iterations: number;
            islands: number;
            sample_size: number;
            migration_interval?: number | undefined;
            map_elites_enabled?: boolean | undefined;
            map_dimensions?: {
                name: string;
                bins: number;
            }[] | undefined;
            temperature_initial?: number | undefined;
            temperature_decay?: number | undefined;
            migration_size?: number | undefined;
            tournament_size?: number | undefined;
            elite_size?: number | undefined;
        }, {
            iterations: number;
            islands: number;
            sample_size: number;
            migration_interval?: number | undefined;
            map_elites_enabled?: boolean | undefined;
            map_dimensions?: {
                name: string;
                bins: number;
            }[] | undefined;
            temperature_initial?: number | undefined;
            temperature_decay?: number | undefined;
            migration_size?: number | undefined;
            tournament_size?: number | undefined;
            elite_size?: number | undefined;
        }>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
        checkpoint_interval: z.ZodOptional<z.ZodNumber>;
        checkpoint_dir: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        evolution: {
            iterations: number;
            islands: number;
            sample_size: number;
            migration_interval?: number | undefined;
            map_elites_enabled?: boolean | undefined;
            map_dimensions?: {
                name: string;
                bins: number;
            }[] | undefined;
            temperature_initial?: number | undefined;
            temperature_decay?: number | undefined;
            migration_size?: number | undefined;
            tournament_size?: number | undefined;
            elite_size?: number | undefined;
        };
        workers: {
            executor: string;
            planner: string;
            summarizer: string;
            planner_config?: Record<string, any> | undefined;
            executor_config?: Record<string, any> | undefined;
            summarizer_config?: Record<string, any> | undefined;
        };
        task_name: string;
        llm_config: {
            timeout: number;
            max_tokens: number;
            model: string;
            context_length: number;
            completion_token_price: number;
            prompt_token_price: number;
            url?: string | undefined;
            api_key?: string | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_provider?: string | undefined;
        };
        timeout_ms?: number | undefined;
        metadata?: Record<string, any> | undefined;
        checkpoint_interval?: number | undefined;
        checkpoint_dir?: string | undefined;
    }, {
        evolution: {
            iterations: number;
            islands: number;
            sample_size: number;
            migration_interval?: number | undefined;
            map_elites_enabled?: boolean | undefined;
            map_dimensions?: {
                name: string;
                bins: number;
            }[] | undefined;
            temperature_initial?: number | undefined;
            temperature_decay?: number | undefined;
            migration_size?: number | undefined;
            tournament_size?: number | undefined;
            elite_size?: number | undefined;
        };
        workers: {
            executor: string;
            planner: string;
            summarizer: string;
            planner_config?: Record<string, any> | undefined;
            executor_config?: Record<string, any> | undefined;
            summarizer_config?: Record<string, any> | undefined;
        };
        task_name: string;
        llm_config: {
            model: string;
            url?: string | undefined;
            timeout?: number | undefined;
            api_key?: string | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            max_tokens?: number | undefined;
            model_provider?: string | undefined;
            context_length?: number | undefined;
            completion_token_price?: number | undefined;
            prompt_token_price?: number | undefined;
        };
        timeout_ms?: number | undefined;
        metadata?: Record<string, any> | undefined;
        checkpoint_interval?: number | undefined;
        checkpoint_dir?: string | undefined;
    }>;
    initial_context: z.ZodRecord<z.ZodString, z.ZodAny>;
    seed_solutions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        solution: z.ZodOptional<z.ZodString>;
        solution_id: z.ZodOptional<z.ZodString>;
        generate_plan: z.ZodOptional<z.ZodString>;
        parent_id: z.ZodOptional<z.ZodString>;
        island_id: z.ZodOptional<z.ZodNumber>;
        iteration: z.ZodOptional<z.ZodNumber>;
        timestamp: z.ZodOptional<z.ZodNumber>;
        generation: z.ZodOptional<z.ZodNumber>;
        sample_cnt: z.ZodOptional<z.ZodNumber>;
        sample_weight: z.ZodOptional<z.ZodNumber>;
        score: z.ZodOptional<z.ZodNumber>;
        evaluation: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        timestamp?: number | undefined;
        metadata?: Record<string, any> | undefined;
        summary?: string | undefined;
        solution?: string | undefined;
        score?: number | undefined;
        evaluation?: string | undefined;
        iteration?: number | undefined;
        island_id?: number | undefined;
        solution_id?: string | undefined;
        parent_id?: string | undefined;
        generate_plan?: string | undefined;
        generation?: number | undefined;
        sample_cnt?: number | undefined;
        sample_weight?: number | undefined;
    }, {
        timestamp?: number | undefined;
        metadata?: Record<string, any> | undefined;
        summary?: string | undefined;
        solution?: string | undefined;
        score?: number | undefined;
        evaluation?: string | undefined;
        iteration?: number | undefined;
        island_id?: number | undefined;
        solution_id?: string | undefined;
        parent_id?: string | undefined;
        generate_plan?: string | undefined;
        generation?: number | undefined;
        sample_cnt?: number | undefined;
        sample_weight?: number | undefined;
    }>, "many">>;
    correlation_id: z.ZodOptional<z.ZodString>;
    created_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    problem_id: string;
    task_config: {
        evolution: {
            iterations: number;
            islands: number;
            sample_size: number;
            migration_interval?: number | undefined;
            map_elites_enabled?: boolean | undefined;
            map_dimensions?: {
                name: string;
                bins: number;
            }[] | undefined;
            temperature_initial?: number | undefined;
            temperature_decay?: number | undefined;
            migration_size?: number | undefined;
            tournament_size?: number | undefined;
            elite_size?: number | undefined;
        };
        workers: {
            executor: string;
            planner: string;
            summarizer: string;
            planner_config?: Record<string, any> | undefined;
            executor_config?: Record<string, any> | undefined;
            summarizer_config?: Record<string, any> | undefined;
        };
        task_name: string;
        llm_config: {
            timeout: number;
            max_tokens: number;
            model: string;
            context_length: number;
            completion_token_price: number;
            prompt_token_price: number;
            url?: string | undefined;
            api_key?: string | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_provider?: string | undefined;
        };
        timeout_ms?: number | undefined;
        metadata?: Record<string, any> | undefined;
        checkpoint_interval?: number | undefined;
        checkpoint_dir?: string | undefined;
    };
    initial_context: Record<string, any>;
    correlation_id?: string | undefined;
    created_at?: string | undefined;
    seed_solutions?: {
        timestamp?: number | undefined;
        metadata?: Record<string, any> | undefined;
        summary?: string | undefined;
        solution?: string | undefined;
        score?: number | undefined;
        evaluation?: string | undefined;
        iteration?: number | undefined;
        island_id?: number | undefined;
        solution_id?: string | undefined;
        parent_id?: string | undefined;
        generate_plan?: string | undefined;
        generation?: number | undefined;
        sample_cnt?: number | undefined;
        sample_weight?: number | undefined;
    }[] | undefined;
}, {
    problem_id: string;
    task_config: {
        evolution: {
            iterations: number;
            islands: number;
            sample_size: number;
            migration_interval?: number | undefined;
            map_elites_enabled?: boolean | undefined;
            map_dimensions?: {
                name: string;
                bins: number;
            }[] | undefined;
            temperature_initial?: number | undefined;
            temperature_decay?: number | undefined;
            migration_size?: number | undefined;
            tournament_size?: number | undefined;
            elite_size?: number | undefined;
        };
        workers: {
            executor: string;
            planner: string;
            summarizer: string;
            planner_config?: Record<string, any> | undefined;
            executor_config?: Record<string, any> | undefined;
            summarizer_config?: Record<string, any> | undefined;
        };
        task_name: string;
        llm_config: {
            model: string;
            url?: string | undefined;
            timeout?: number | undefined;
            api_key?: string | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            max_tokens?: number | undefined;
            model_provider?: string | undefined;
            context_length?: number | undefined;
            completion_token_price?: number | undefined;
            prompt_token_price?: number | undefined;
        };
        timeout_ms?: number | undefined;
        metadata?: Record<string, any> | undefined;
        checkpoint_interval?: number | undefined;
        checkpoint_dir?: string | undefined;
    };
    initial_context: Record<string, any>;
    correlation_id?: string | undefined;
    created_at?: string | undefined;
    seed_solutions?: {
        timestamp?: number | undefined;
        metadata?: Record<string, any> | undefined;
        summary?: string | undefined;
        solution?: string | undefined;
        score?: number | undefined;
        evaluation?: string | undefined;
        iteration?: number | undefined;
        island_id?: number | undefined;
        solution_id?: string | undefined;
        parent_id?: string | undefined;
        generate_plan?: string | undefined;
        generation?: number | undefined;
        sample_cnt?: number | undefined;
        sample_weight?: number | undefined;
    }[] | undefined;
}>;
export type LoongFlowRequest = z.infer<typeof LoongFlowRequest>;
/**
 * LoongFlow Response Schema
 *
 * Represents a response from LoongFlow execution.
 */
export declare const LoongFlowResponse: z.ZodObject<{
    request_id: z.ZodString;
    state: z.ZodEnum<["idle", "planning", "executing", "summarizing", "evolving", "completed", "failed", "cancelled"]>;
    current_solution: z.ZodOptional<z.ZodObject<{
        solution: z.ZodString;
        solution_id: z.ZodString;
        generate_plan: z.ZodString;
        parent_id: z.ZodString;
        island_id: z.ZodNumber;
        iteration: z.ZodNumber;
        timestamp: z.ZodNumber;
        generation: z.ZodNumber;
        sample_cnt: z.ZodNumber;
        sample_weight: z.ZodNumber;
        score: z.ZodNumber;
        evaluation: z.ZodString;
        summary: z.ZodString;
        metadata: z.ZodRecord<z.ZodString, z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }, {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }>>;
    best_solutions: z.ZodArray<z.ZodObject<{
        solution: z.ZodString;
        solution_id: z.ZodString;
        generate_plan: z.ZodString;
        parent_id: z.ZodString;
        island_id: z.ZodNumber;
        iteration: z.ZodNumber;
        timestamp: z.ZodNumber;
        generation: z.ZodNumber;
        sample_cnt: z.ZodNumber;
        sample_weight: z.ZodNumber;
        score: z.ZodNumber;
        evaluation: z.ZodString;
        summary: z.ZodString;
        metadata: z.ZodRecord<z.ZodString, z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }, {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }>, "many">;
    iteration: z.ZodNumber;
    checkpoints: z.ZodArray<z.ZodString, "many">;
    metrics: z.ZodOptional<z.ZodObject<{
        total_duration_ms: z.ZodOptional<z.ZodNumber>;
        planning_time_ms: z.ZodOptional<z.ZodNumber>;
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        summarization_time_ms: z.ZodOptional<z.ZodNumber>;
        solutions_generated: z.ZodOptional<z.ZodNumber>;
        islands_completed: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        execution_time_ms?: number | undefined;
        total_duration_ms?: number | undefined;
        planning_time_ms?: number | undefined;
        summarization_time_ms?: number | undefined;
        solutions_generated?: number | undefined;
        islands_completed?: number | undefined;
    }, {
        execution_time_ms?: number | undefined;
        total_duration_ms?: number | undefined;
        planning_time_ms?: number | undefined;
        summarization_time_ms?: number | undefined;
        solutions_generated?: number | undefined;
        islands_completed?: number | undefined;
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
    timestamp: z.ZodOptional<z.ZodString>;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    state: "idle" | "completed" | "failed" | "cancelled" | "planning" | "executing" | "summarizing" | "evolving";
    request_id: string;
    iteration: number;
    best_solutions: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }[];
    checkpoints: string[];
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    timestamp?: string | undefined;
    metadata?: Record<string, any> | undefined;
    metrics?: {
        execution_time_ms?: number | undefined;
        total_duration_ms?: number | undefined;
        planning_time_ms?: number | undefined;
        summarization_time_ms?: number | undefined;
        solutions_generated?: number | undefined;
        islands_completed?: number | undefined;
    } | undefined;
    current_solution?: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    } | undefined;
}, {
    state: "idle" | "completed" | "failed" | "cancelled" | "planning" | "executing" | "summarizing" | "evolving";
    request_id: string;
    iteration: number;
    best_solutions: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }[];
    checkpoints: string[];
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    timestamp?: string | undefined;
    metadata?: Record<string, any> | undefined;
    metrics?: {
        execution_time_ms?: number | undefined;
        total_duration_ms?: number | undefined;
        planning_time_ms?: number | undefined;
        summarization_time_ms?: number | undefined;
        solutions_generated?: number | undefined;
        islands_completed?: number | undefined;
    } | undefined;
    current_solution?: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    } | undefined;
}>;
export type LoongFlowResponse = z.infer<typeof LoongFlowResponse>;
/**
 * LoongFlow Checkpoint Schema
 *
 * Represents a checkpoint in LoongFlow execution.
 */
export declare const LoongFlowCheckpoint: z.ZodObject<{
    checkpoint_id: z.ZodString;
    iteration: z.ZodNumber;
    state: z.ZodEnum<["idle", "planning", "executing", "summarizing", "evolving", "completed", "failed", "cancelled"]>;
    populations: z.ZodArray<z.ZodObject<{
        island_id: z.ZodNumber;
        solutions: z.ZodArray<z.ZodObject<{
            solution: z.ZodString;
            solution_id: z.ZodString;
            generate_plan: z.ZodString;
            parent_id: z.ZodString;
            island_id: z.ZodNumber;
            iteration: z.ZodNumber;
            timestamp: z.ZodNumber;
            generation: z.ZodNumber;
            sample_cnt: z.ZodNumber;
            sample_weight: z.ZodNumber;
            score: z.ZodNumber;
            evaluation: z.ZodString;
            summary: z.ZodString;
            metadata: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        solutions: {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[];
        island_id: number;
    }, {
        solutions: {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[];
        island_id: number;
    }>, "many">;
    fitness_maps: z.ZodOptional<z.ZodArray<z.ZodObject<{
        island_id: z.ZodNumber;
        map: z.ZodRecord<z.ZodString, z.ZodArray<z.ZodObject<{
            solution: z.ZodString;
            solution_id: z.ZodString;
            generate_plan: z.ZodString;
            parent_id: z.ZodString;
            island_id: z.ZodNumber;
            iteration: z.ZodNumber;
            timestamp: z.ZodNumber;
            generation: z.ZodNumber;
            sample_cnt: z.ZodNumber;
            sample_weight: z.ZodNumber;
            score: z.ZodNumber;
            evaluation: z.ZodString;
            summary: z.ZodString;
            metadata: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        map: Record<string, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[]>;
        island_id: number;
    }, {
        map: Record<string, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[]>;
        island_id: number;
    }>, "many">>;
    timestamp: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    state: "idle" | "completed" | "failed" | "cancelled" | "planning" | "executing" | "summarizing" | "evolving";
    iteration: number;
    checkpoint_id: string;
    populations: {
        solutions: {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[];
        island_id: number;
    }[];
    metadata?: Record<string, any> | undefined;
    fitness_maps?: {
        map: Record<string, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[]>;
        island_id: number;
    }[] | undefined;
}, {
    timestamp: string;
    state: "idle" | "completed" | "failed" | "cancelled" | "planning" | "executing" | "summarizing" | "evolving";
    iteration: number;
    checkpoint_id: string;
    populations: {
        solutions: {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[];
        island_id: number;
    }[];
    metadata?: Record<string, any> | undefined;
    fitness_maps?: {
        map: Record<string, {
            timestamp: number;
            metadata: Record<string, any>;
            summary: string;
            solution: string;
            score: number;
            evaluation: string;
            iteration: number;
            island_id: number;
            solution_id: string;
            parent_id: string;
            generate_plan: string;
            generation: number;
            sample_cnt: number;
            sample_weight: number;
        }[]>;
        island_id: number;
    }[] | undefined;
}>;
export type LoongFlowCheckpoint = z.infer<typeof LoongFlowCheckpoint>;
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform raw LoongFlow solution to canonical format
 */
export declare function transformLoongFlowSolutionToCanonical(rawSolution: any): LoongFlowSolution;
/**
 * Transform canonical LoongFlow solution to raw format
 */
export declare function transformCanonicalToLoongFlowSolution(canonical: LoongFlowSolution): Record<string, any>;
/**
 * Transform LoongFlow response to canonical ExecutionResult
 */
export declare function transformLoongFlowResponseToExecutionResult(lfResponse: LoongFlowResponse, problemId: string, planId: string): ExecutionResultType;
/**
 * Transform canonical Problem to LoongFlow request
 */
export declare function transformCanonicalProblemToLoongFlowRequest(problem: ProblemType, config: LoongFlowConfig, initialContext?: Record<string, any>): LoongFlowRequest;
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate a LoongFlowSolution object
 */
export declare function validateLoongFlowSolution(data: unknown): {
    success: boolean;
    data?: LoongFlowSolution;
    errors?: string[];
};
/**
 * Validate a LoongFlowConfig object
 */
export declare function validateLoongFlowConfig(data: unknown): {
    success: boolean;
    data?: LoongFlowConfig;
    errors?: string[];
};
/**
 * Validate a LoongFlowRequest object
 */
export declare function validateLoongFlowRequest(data: unknown): {
    success: boolean;
    data?: LoongFlowRequest;
    errors?: string[];
};
/**
 * Validate a LoongFlowResponse object
 */
export declare function validateLoongFlowResponse(data: unknown): {
    success: boolean;
    data?: LoongFlowResponse;
    errors?: string[];
};
/**
 * ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================
 */
/**
 * Check if data is a valid LoongFlowSolution
 */
export declare function isLoongFlowSolution(data: unknown): data is LoongFlowSolution;
/**
 * Check if data is a valid LoongFlowRequest
 */
export declare function isLoongFlowRequest(data: unknown): data is LoongFlowRequest;
/**
 * Check if data is a valid LoongFlowResponse
 */
export declare function isLoongFlowResponse(data: unknown): data is LoongFlowResponse;
/**
 * ============================================================================
 * EXAMPLES
 * ============================================================================
 */
/**
 * Example usage:
 * @example
 * ```typescript
 * import {
 *   validateLoongFlowRequest,
 *   createPESUTCTimestamp,
 *   createPESCorrelationId
 * } from './loongflow-canonical';
 *
 * const request = {
 *   problem_id: createPESCorrelationId(),
 *   task_config: {
 *     task_name: 'Optimize neural network architecture',
 *     llm_config: {
 *       provider: 'openai',
 *       model: 'gpt-4',
 *       temperature: 0.7,
 *       max_tokens: 2000,
 *     },
 *     workers: {
 *       planner: 'LLMPlanner',
 *       executor: 'CodeExecutor',
 *       summarizer: 'QualitySummarizer',
 *     },
 *     evolution: {
 *       iterations: 10,
 *       islands: 4,
 *       sample_size: 100,
 *     },
 *   },
 *   initial_context: {
 *     dataset: 'MNIST',
 *     max_layers: 10,
 *   },
 *   created_at: createPESUTCTimestamp(),
 * };
 *
 * const validation = validateLoongFlowRequest(request);
 * if (!validation.success) {
 *   console.error('Invalid request:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=loongflow-canonical.d.ts.map