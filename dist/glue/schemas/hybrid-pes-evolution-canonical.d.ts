/**
 * Hybrid PES-Evolution Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for hybrid workflows that combine
 * Plan-Execute-Summarize (PES) with Evolutionary Optimization paradigms.
 *
 * The hybrid approach enables:
 * - PES_OPTIMIZE: Use PES to plan, then evolutionary algorithms to optimize
 * - EVOLVE_SOLVE: Use evolution to explore solution space, then PES to refine
 * - ADAPTIVE_EXECUTE: Dynamically switch between paradigms based on performance
 *
 * This schema bridges LoongFlow (PES + Evolution) with OpenEvolve (pure evolution)
 * to enable powerful hybrid problem-solving strategies.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for hybrid workflow
 * data in the glue layer. Do not pass raw API responses between services.
 */
import { z } from 'zod';
import { type Summary as SummaryType } from './pes-canonical';
import { type LoongFlowSolution as LoongFlowSolutionType } from './loongflow-canonical';
/**
 * Hybrid Task Type Enum
 *
 * Defines the type of hybrid task.
 */
export declare const HybridTaskType: z.ZodEnum<["pes_optimize", "evolve_solve", "adaptive_execute", "parallel_hybrid", "interleaved"]>;
export type HybridTaskType = z.infer<typeof HybridTaskType>;
/**
 * Integration Strategy Enum
 *
 * Defines how PES and Evolution are integrated.
 */
export declare const IntegrationStrategy: z.ZodEnum<["sequential", "parallel", "interleaved", "adaptive"]>;
export type IntegrationStrategy = z.infer<typeof IntegrationStrategy>;
/**
 * Adaptive Trigger Condition Enum
 *
 * Conditions that trigger adaptive behavior changes.
 */
export declare const AdaptiveTriggerCondition: z.ZodEnum<["low_confidence", "high_complexity", "stagnation", "timeout_risk", "convergence_detected", "divergence_detected"]>;
export type AdaptiveTriggerCondition = z.infer<typeof AdaptiveTriggerCondition>;
/**
 * Adaptive Action Enum
 *
 * Actions to take when trigger condition is met.
 */
export declare const AdaptiveAction: z.ZodEnum<["switch_to_evolution", "switch_to_pes", "increase_iterations", "call_for_help", "merge_populations", "reset_and_restart"]>;
export type AdaptiveAction = z.infer<typeof AdaptiveAction>;
/**
 * Knowledge Source Type Enum
 *
 * Types of evolutionary knowledge sources.
 */
export declare const KnowledgeSourceType: z.ZodEnum<["loongflow_solution", "openevolve_optimization", "hybrid_result", "external_solution"]>;
export type KnowledgeSourceType = z.infer<typeof KnowledgeSourceType>;
/**
 * Knowledge Type Enum
 *
 * Types of knowledge that can be extracted.
 */
export declare const KnowledgeType: z.ZodEnum<["solution_pattern", "planning_strategy", "execution_approach", "parameter_setting"]>;
export type KnowledgeType = z.infer<typeof KnowledgeType>;
/**
 * ============================================================================
 * HYBRID TASK SCHEMAS
 * ============================================================================
 */
/**
 * Evolution Configuration Schema
 *
 * Generic evolutionary algorithm configuration (for OpenEvolve).
 */
export declare const EvolutionConfig: z.ZodObject<{
    generations: z.ZodNumber;
    population_size: z.ZodNumber;
    mutation_rate: z.ZodNumber;
    crossover_rate: z.ZodNumber;
    selection_method: z.ZodOptional<z.ZodString>;
    elitism_count: z.ZodOptional<z.ZodNumber>;
    mutation_strength: z.ZodOptional<z.ZodNumber>;
    representation_type: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    population_size: number;
    generations: number;
    mutation_rate: number;
    crossover_rate: number;
    metadata?: Record<string, any> | undefined;
    selection_method?: string | undefined;
    elitism_count?: number | undefined;
    mutation_strength?: number | undefined;
    representation_type?: string | undefined;
}, {
    population_size: number;
    generations: number;
    mutation_rate: number;
    crossover_rate: number;
    metadata?: Record<string, any> | undefined;
    selection_method?: string | undefined;
    elitism_count?: number | undefined;
    mutation_strength?: number | undefined;
    representation_type?: string | undefined;
}>;
export type EvolutionConfig = z.infer<typeof EvolutionConfig>;
/**
 * Hybrid Task Schema
 *
 * Represents a task that combines PES and Evolution.
 */
export declare const HybridTask: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["pes_optimize", "evolve_solve", "adaptive_execute", "parallel_hybrid", "interleaved"]>;
    problem: z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["optimization", "reasoning", "generation", "validation", "classification", "prediction", "synthesis"]>;
        description: z.ZodString;
        context: z.ZodRecord<z.ZodString, z.ZodAny>;
        constraints: z.ZodArray<z.ZodString, "many">;
        success_criteria: z.ZodArray<z.ZodString, "many">;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        created_at: z.ZodString;
        priority: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: "classification" | "validation" | "synthesis" | "optimization" | "reasoning" | "generation" | "prediction";
        id: string;
        constraints: string[];
        created_at: string;
        description: string;
        context: Record<string, any>;
        success_criteria: string[];
        metadata?: Record<string, any> | undefined;
        priority?: number | undefined;
        tags?: string[] | undefined;
    }, {
        type: "classification" | "validation" | "synthesis" | "optimization" | "reasoning" | "generation" | "prediction";
        id: string;
        constraints: string[];
        created_at: string;
        description: string;
        context: Record<string, any>;
        success_criteria: string[];
        metadata?: Record<string, any> | undefined;
        priority?: number | undefined;
        tags?: string[] | undefined;
    }>;
    pes_config: z.ZodOptional<z.ZodObject<{
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
    }>>;
    evolution_config: z.ZodOptional<z.ZodObject<{
        generations: z.ZodNumber;
        population_size: z.ZodNumber;
        mutation_rate: z.ZodNumber;
        crossover_rate: z.ZodNumber;
        selection_method: z.ZodOptional<z.ZodString>;
        elitism_count: z.ZodOptional<z.ZodNumber>;
        mutation_strength: z.ZodOptional<z.ZodNumber>;
        representation_type: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        population_size: number;
        generations: number;
        mutation_rate: number;
        crossover_rate: number;
        metadata?: Record<string, any> | undefined;
        selection_method?: string | undefined;
        elitism_count?: number | undefined;
        mutation_strength?: number | undefined;
        representation_type?: string | undefined;
    }, {
        population_size: number;
        generations: number;
        mutation_rate: number;
        crossover_rate: number;
        metadata?: Record<string, any> | undefined;
        selection_method?: string | undefined;
        elitism_count?: number | undefined;
        mutation_strength?: number | undefined;
        representation_type?: string | undefined;
    }>>;
    integration_strategy: z.ZodEnum<["sequential", "parallel", "interleaved", "adaptive"]>;
    adaptive_triggers: z.ZodOptional<z.ZodArray<z.ZodObject<{
        condition: z.ZodEnum<["low_confidence", "high_complexity", "stagnation", "timeout_risk", "convergence_detected", "divergence_detected"]>;
        threshold: z.ZodNumber;
        action: z.ZodEnum<["switch_to_evolution", "switch_to_pes", "increase_iterations", "call_for_help", "merge_populations", "reset_and_restart"]>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        threshold: number;
        action: "switch_to_evolution" | "switch_to_pes" | "increase_iterations" | "call_for_help" | "merge_populations" | "reset_and_restart";
        condition: "low_confidence" | "high_complexity" | "stagnation" | "timeout_risk" | "convergence_detected" | "divergence_detected";
        metadata?: Record<string, any> | undefined;
    }, {
        threshold: number;
        action: "switch_to_evolution" | "switch_to_pes" | "increase_iterations" | "call_for_help" | "merge_populations" | "reset_and_restart";
        condition: "low_confidence" | "high_complexity" | "stagnation" | "timeout_risk" | "convergence_detected" | "divergence_detected";
        metadata?: Record<string, any> | undefined;
    }>, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
    priority: z.ZodOptional<z.ZodNumber>;
    timeout_ms: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    type: "pes_optimize" | "evolve_solve" | "adaptive_execute" | "parallel_hybrid" | "interleaved";
    id: string;
    problem: {
        type: "classification" | "validation" | "synthesis" | "optimization" | "reasoning" | "generation" | "prediction";
        id: string;
        constraints: string[];
        created_at: string;
        description: string;
        context: Record<string, any>;
        success_criteria: string[];
        metadata?: Record<string, any> | undefined;
        priority?: number | undefined;
        tags?: string[] | undefined;
    };
    created_at: string;
    integration_strategy: "adaptive" | "sequential" | "parallel" | "interleaved";
    correlation_id?: string | undefined;
    timeout_ms?: number | undefined;
    metadata?: Record<string, any> | undefined;
    priority?: number | undefined;
    pes_config?: {
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
    } | undefined;
    evolution_config?: {
        population_size: number;
        generations: number;
        mutation_rate: number;
        crossover_rate: number;
        metadata?: Record<string, any> | undefined;
        selection_method?: string | undefined;
        elitism_count?: number | undefined;
        mutation_strength?: number | undefined;
        representation_type?: string | undefined;
    } | undefined;
    adaptive_triggers?: {
        threshold: number;
        action: "switch_to_evolution" | "switch_to_pes" | "increase_iterations" | "call_for_help" | "merge_populations" | "reset_and_restart";
        condition: "low_confidence" | "high_complexity" | "stagnation" | "timeout_risk" | "convergence_detected" | "divergence_detected";
        metadata?: Record<string, any> | undefined;
    }[] | undefined;
}, {
    type: "pes_optimize" | "evolve_solve" | "adaptive_execute" | "parallel_hybrid" | "interleaved";
    id: string;
    problem: {
        type: "classification" | "validation" | "synthesis" | "optimization" | "reasoning" | "generation" | "prediction";
        id: string;
        constraints: string[];
        created_at: string;
        description: string;
        context: Record<string, any>;
        success_criteria: string[];
        metadata?: Record<string, any> | undefined;
        priority?: number | undefined;
        tags?: string[] | undefined;
    };
    created_at: string;
    integration_strategy: "adaptive" | "sequential" | "parallel" | "interleaved";
    correlation_id?: string | undefined;
    timeout_ms?: number | undefined;
    metadata?: Record<string, any> | undefined;
    priority?: number | undefined;
    pes_config?: {
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
    } | undefined;
    evolution_config?: {
        population_size: number;
        generations: number;
        mutation_rate: number;
        crossover_rate: number;
        metadata?: Record<string, any> | undefined;
        selection_method?: string | undefined;
        elitism_count?: number | undefined;
        mutation_strength?: number | undefined;
        representation_type?: string | undefined;
    } | undefined;
    adaptive_triggers?: {
        threshold: number;
        action: "switch_to_evolution" | "switch_to_pes" | "increase_iterations" | "call_for_help" | "merge_populations" | "reset_and_restart";
        condition: "low_confidence" | "high_complexity" | "stagnation" | "timeout_risk" | "convergence_detected" | "divergence_detected";
        metadata?: Record<string, any> | undefined;
    }[] | undefined;
}>;
export type HybridTask = z.infer<typeof HybridTask>;
/**
 * Evolutionary Knowledge Schema
 *
 * Represents knowledge extracted from evolutionary processes.
 */
export declare const EvolutionaryKnowledge: z.ZodObject<{
    id: z.ZodString;
    source_type: z.ZodEnum<["loongflow_solution", "openevolve_optimization", "hybrid_result", "external_solution"]>;
    problem_id: z.ZodString;
    knowledge_type: z.ZodEnum<["solution_pattern", "planning_strategy", "execution_approach", "parameter_setting"]>;
    content: z.ZodObject<{
        pattern: z.ZodString;
        success_rate: z.ZodNumber;
        avg_score: z.ZodNumber;
        usage_count: z.ZodNumber;
        context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        pattern: string;
        success_rate: number;
        avg_score: number;
        usage_count: number;
        context?: Record<string, any> | undefined;
    }, {
        pattern: string;
        success_rate: number;
        avg_score: number;
        usage_count: number;
        context?: Record<string, any> | undefined;
    }>;
    source_id: z.ZodOptional<z.ZodString>;
    valid_until: z.ZodOptional<z.ZodString>;
    extracted_at: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    id: string;
    content: {
        pattern: string;
        success_rate: number;
        avg_score: number;
        usage_count: number;
        context?: Record<string, any> | undefined;
    };
    source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
    problem_id: string;
    knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
    extracted_at: string;
    metadata?: Record<string, any> | undefined;
    source_id?: string | undefined;
    valid_until?: string | undefined;
}, {
    id: string;
    content: {
        pattern: string;
        success_rate: number;
        avg_score: number;
        usage_count: number;
        context?: Record<string, any> | undefined;
    };
    source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
    problem_id: string;
    knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
    extracted_at: string;
    metadata?: Record<string, any> | undefined;
    source_id?: string | undefined;
    valid_until?: string | undefined;
}>;
export type EvolutionaryKnowledge = z.infer<typeof EvolutionaryKnowledge>;
/**
 * Population Individual Schema
 *
 * Represents an individual in an evolutionary population.
 */
export declare const PopulationIndividual: z.ZodObject<{
    id: z.ZodString;
    genes: z.ZodAny;
    fitness: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    parent_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    generation: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    id: string;
    fitness: number;
    metadata?: Record<string, any> | undefined;
    generation?: number | undefined;
    genes?: any;
    parent_ids?: string[] | undefined;
}, {
    id: string;
    fitness: number;
    metadata?: Record<string, any> | undefined;
    generation?: number | undefined;
    genes?: any;
    parent_ids?: string[] | undefined;
}>;
export type PopulationIndividual = z.infer<typeof PopulationIndividual>;
/**
 * Evolution Result Schema
 *
 * Represents results from an evolutionary optimization run.
 */
export declare const EvolutionResult: z.ZodObject<{
    best_fitness: z.ZodNumber;
    generations: z.ZodNumber;
    population: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        genes: z.ZodAny;
        fitness: z.ZodNumber;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        parent_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        generation: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    }, {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    }>, "many">;
    converged: z.ZodOptional<z.ZodBoolean>;
    convergence_generation: z.ZodOptional<z.ZodNumber>;
    diversity_score: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    best_fitness: number;
    generations: number;
    population: {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    }[];
    metadata?: Record<string, any> | undefined;
    converged?: boolean | undefined;
    convergence_generation?: number | undefined;
    diversity_score?: number | undefined;
}, {
    best_fitness: number;
    generations: number;
    population: {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    }[];
    metadata?: Record<string, any> | undefined;
    converged?: boolean | undefined;
    convergence_generation?: number | undefined;
    diversity_score?: number | undefined;
}>;
export type EvolutionResult = z.infer<typeof EvolutionResult>;
/**
 * Integration Metrics Schema
 *
 * Metrics for hybrid workflow integration.
 */
export declare const IntegrationMetrics: z.ZodObject<{
    pes_iterations: z.ZodNumber;
    evolution_generations: z.ZodNumber;
    total_duration_ms: z.ZodNumber;
    synergy_score: z.ZodNumber;
    pes_time_ms: z.ZodOptional<z.ZodNumber>;
    evolution_time_ms: z.ZodOptional<z.ZodNumber>;
    paradigm_switches: z.ZodOptional<z.ZodNumber>;
    knowledge_transferred: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    total_duration_ms: number;
    pes_iterations: number;
    evolution_generations: number;
    synergy_score: number;
    pes_time_ms?: number | undefined;
    evolution_time_ms?: number | undefined;
    paradigm_switches?: number | undefined;
    knowledge_transferred?: number | undefined;
}, {
    total_duration_ms: number;
    pes_iterations: number;
    evolution_generations: number;
    synergy_score: number;
    pes_time_ms?: number | undefined;
    evolution_time_ms?: number | undefined;
    paradigm_switches?: number | undefined;
    knowledge_transferred?: number | undefined;
}>;
export type IntegrationMetrics = z.infer<typeof IntegrationMetrics>;
/**
 * Hybrid Execution Result Schema
 *
 * Represents the result of a hybrid workflow execution.
 */
export declare const HybridExecutionResult: z.ZodObject<{
    id: z.ZodString;
    task_id: z.ZodString;
    pes_result: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        problem_id: z.ZodString;
        plan_id: z.ZodString;
        state: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "timeout", "paused"]>;
        outputs: z.ZodRecord<z.ZodString, z.ZodAny>;
        metrics: z.ZodObject<{
            duration_ms: z.ZodNumber;
            success: z.ZodBoolean;
            score: z.ZodOptional<z.ZodNumber>;
            error_count: z.ZodNumber;
            steps_completed: z.ZodOptional<z.ZodNumber>;
            steps_failed: z.ZodOptional<z.ZodNumber>;
            memory_peak_mb: z.ZodOptional<z.ZodNumber>;
            cpu_time_ms: z.ZodOptional<z.ZodNumber>;
            iterations: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            duration_ms: number;
            success: boolean;
            error_count: number;
            steps_completed?: number | undefined;
            steps_failed?: number | undefined;
            score?: number | undefined;
            iterations?: number | undefined;
            memory_peak_mb?: number | undefined;
            cpu_time_ms?: number | undefined;
        }, {
            duration_ms: number;
            success: boolean;
            error_count: number;
            steps_completed?: number | undefined;
            steps_failed?: number | undefined;
            score?: number | undefined;
            iterations?: number | undefined;
            memory_peak_mb?: number | undefined;
            cpu_time_ms?: number | undefined;
        }>;
        artifacts: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodString;
            uri: z.ZodString;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            size_bytes: z.ZodOptional<z.ZodNumber>;
            checksum: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            uri: string;
            metadata?: Record<string, any> | undefined;
            size_bytes?: number | undefined;
            checksum?: string | undefined;
        }, {
            type: string;
            uri: string;
            metadata?: Record<string, any> | undefined;
            size_bytes?: number | undefined;
            checksum?: string | undefined;
        }>, "many">>;
        logs: z.ZodOptional<z.ZodArray<z.ZodObject<{
            timestamp: z.ZodString;
            level: z.ZodEnum<["debug", "info", "warn", "error", "fatal"]>;
            message: z.ZodString;
            context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            correlation_id: z.ZodOptional<z.ZodString>;
            step_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            timestamp: string;
            level: "debug" | "info" | "warn" | "error" | "fatal";
            message: string;
            correlation_id?: string | undefined;
            step_id?: string | undefined;
            context?: Record<string, any> | undefined;
        }, {
            timestamp: string;
            level: "debug" | "info" | "warn" | "error" | "fatal";
            message: string;
            correlation_id?: string | undefined;
            step_id?: string | undefined;
            context?: Record<string, any> | undefined;
        }>, "many">>;
        error: z.ZodOptional<z.ZodObject<{
            code: z.ZodString;
            message: z.ZodString;
            details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            stack_trace: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
            stack_trace?: string | undefined;
        }, {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
            stack_trace?: string | undefined;
        }>>;
        created_at: z.ZodString;
        completed_at: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        correlation_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        state: "running" | "completed" | "failed" | "pending" | "cancelled" | "timeout" | "paused";
        metrics: {
            duration_ms: number;
            success: boolean;
            error_count: number;
            steps_completed?: number | undefined;
            steps_failed?: number | undefined;
            score?: number | undefined;
            iterations?: number | undefined;
            memory_peak_mb?: number | undefined;
            cpu_time_ms?: number | undefined;
        };
        id: string;
        created_at: string;
        problem_id: string;
        plan_id: string;
        outputs: Record<string, any>;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
            stack_trace?: string | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        completed_at?: string | undefined;
        logs?: {
            timestamp: string;
            level: "debug" | "info" | "warn" | "error" | "fatal";
            message: string;
            correlation_id?: string | undefined;
            step_id?: string | undefined;
            context?: Record<string, any> | undefined;
        }[] | undefined;
        artifacts?: {
            type: string;
            uri: string;
            metadata?: Record<string, any> | undefined;
            size_bytes?: number | undefined;
            checksum?: string | undefined;
        }[] | undefined;
    }, {
        state: "running" | "completed" | "failed" | "pending" | "cancelled" | "timeout" | "paused";
        metrics: {
            duration_ms: number;
            success: boolean;
            error_count: number;
            steps_completed?: number | undefined;
            steps_failed?: number | undefined;
            score?: number | undefined;
            iterations?: number | undefined;
            memory_peak_mb?: number | undefined;
            cpu_time_ms?: number | undefined;
        };
        id: string;
        created_at: string;
        problem_id: string;
        plan_id: string;
        outputs: Record<string, any>;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
            stack_trace?: string | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        completed_at?: string | undefined;
        logs?: {
            timestamp: string;
            level: "debug" | "info" | "warn" | "error" | "fatal";
            message: string;
            correlation_id?: string | undefined;
            step_id?: string | undefined;
            context?: Record<string, any> | undefined;
        }[] | undefined;
        artifacts?: {
            type: string;
            uri: string;
            metadata?: Record<string, any> | undefined;
            size_bytes?: number | undefined;
            checksum?: string | undefined;
        }[] | undefined;
    }>>;
    evolution_result: z.ZodOptional<z.ZodObject<{
        best_fitness: z.ZodNumber;
        generations: z.ZodNumber;
        population: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            genes: z.ZodAny;
            fitness: z.ZodNumber;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            parent_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            generation: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            fitness: number;
            metadata?: Record<string, any> | undefined;
            generation?: number | undefined;
            genes?: any;
            parent_ids?: string[] | undefined;
        }, {
            id: string;
            fitness: number;
            metadata?: Record<string, any> | undefined;
            generation?: number | undefined;
            genes?: any;
            parent_ids?: string[] | undefined;
        }>, "many">;
        converged: z.ZodOptional<z.ZodBoolean>;
        convergence_generation: z.ZodOptional<z.ZodNumber>;
        diversity_score: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        best_fitness: number;
        generations: number;
        population: {
            id: string;
            fitness: number;
            metadata?: Record<string, any> | undefined;
            generation?: number | undefined;
            genes?: any;
            parent_ids?: string[] | undefined;
        }[];
        metadata?: Record<string, any> | undefined;
        converged?: boolean | undefined;
        convergence_generation?: number | undefined;
        diversity_score?: number | undefined;
    }, {
        best_fitness: number;
        generations: number;
        population: {
            id: string;
            fitness: number;
            metadata?: Record<string, any> | undefined;
            generation?: number | undefined;
            genes?: any;
            parent_ids?: string[] | undefined;
        }[];
        metadata?: Record<string, any> | undefined;
        converged?: boolean | undefined;
        convergence_generation?: number | undefined;
        diversity_score?: number | undefined;
    }>>;
    integration_metrics: z.ZodObject<{
        pes_iterations: z.ZodNumber;
        evolution_generations: z.ZodNumber;
        total_duration_ms: z.ZodNumber;
        synergy_score: z.ZodNumber;
        pes_time_ms: z.ZodOptional<z.ZodNumber>;
        evolution_time_ms: z.ZodOptional<z.ZodNumber>;
        paradigm_switches: z.ZodOptional<z.ZodNumber>;
        knowledge_transferred: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_duration_ms: number;
        pes_iterations: number;
        evolution_generations: number;
        synergy_score: number;
        pes_time_ms?: number | undefined;
        evolution_time_ms?: number | undefined;
        paradigm_switches?: number | undefined;
        knowledge_transferred?: number | undefined;
    }, {
        total_duration_ms: number;
        pes_iterations: number;
        evolution_generations: number;
        synergy_score: number;
        pes_time_ms?: number | undefined;
        evolution_time_ms?: number | undefined;
        paradigm_switches?: number | undefined;
        knowledge_transferred?: number | undefined;
    }>;
    best_solution: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
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
    }>, z.ZodObject<{
        id: z.ZodString;
        genes: z.ZodAny;
        fitness: z.ZodNumber;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        parent_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        generation: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    }, {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    }>]>>;
    summary: z.ZodOptional<z.ZodString>;
    knowledge_extracted: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        source_type: z.ZodEnum<["loongflow_solution", "openevolve_optimization", "hybrid_result", "external_solution"]>;
        problem_id: z.ZodString;
        knowledge_type: z.ZodEnum<["solution_pattern", "planning_strategy", "execution_approach", "parameter_setting"]>;
        content: z.ZodObject<{
            pattern: z.ZodString;
            success_rate: z.ZodNumber;
            avg_score: z.ZodNumber;
            usage_count: z.ZodNumber;
            context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        }, {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        }>;
        source_id: z.ZodOptional<z.ZodString>;
        valid_until: z.ZodOptional<z.ZodString>;
        extracted_at: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    }, {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    }>, "many">>;
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
    created_at: z.ZodString;
    completed_at: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    created_at: string;
    task_id: string;
    integration_metrics: {
        total_duration_ms: number;
        pes_iterations: number;
        evolution_generations: number;
        synergy_score: number;
        pes_time_ms?: number | undefined;
        evolution_time_ms?: number | undefined;
        paradigm_switches?: number | undefined;
        knowledge_transferred?: number | undefined;
    };
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    summary?: string | undefined;
    completed_at?: string | undefined;
    best_solution?: {
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
    } | {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    } | undefined;
    pes_result?: {
        state: "running" | "completed" | "failed" | "pending" | "cancelled" | "timeout" | "paused";
        metrics: {
            duration_ms: number;
            success: boolean;
            error_count: number;
            steps_completed?: number | undefined;
            steps_failed?: number | undefined;
            score?: number | undefined;
            iterations?: number | undefined;
            memory_peak_mb?: number | undefined;
            cpu_time_ms?: number | undefined;
        };
        id: string;
        created_at: string;
        problem_id: string;
        plan_id: string;
        outputs: Record<string, any>;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
            stack_trace?: string | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        completed_at?: string | undefined;
        logs?: {
            timestamp: string;
            level: "debug" | "info" | "warn" | "error" | "fatal";
            message: string;
            correlation_id?: string | undefined;
            step_id?: string | undefined;
            context?: Record<string, any> | undefined;
        }[] | undefined;
        artifacts?: {
            type: string;
            uri: string;
            metadata?: Record<string, any> | undefined;
            size_bytes?: number | undefined;
            checksum?: string | undefined;
        }[] | undefined;
    } | undefined;
    evolution_result?: {
        best_fitness: number;
        generations: number;
        population: {
            id: string;
            fitness: number;
            metadata?: Record<string, any> | undefined;
            generation?: number | undefined;
            genes?: any;
            parent_ids?: string[] | undefined;
        }[];
        metadata?: Record<string, any> | undefined;
        converged?: boolean | undefined;
        convergence_generation?: number | undefined;
        diversity_score?: number | undefined;
    } | undefined;
    knowledge_extracted?: {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    }[] | undefined;
}, {
    id: string;
    created_at: string;
    task_id: string;
    integration_metrics: {
        total_duration_ms: number;
        pes_iterations: number;
        evolution_generations: number;
        synergy_score: number;
        pes_time_ms?: number | undefined;
        evolution_time_ms?: number | undefined;
        paradigm_switches?: number | undefined;
        knowledge_transferred?: number | undefined;
    };
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    summary?: string | undefined;
    completed_at?: string | undefined;
    best_solution?: {
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
    } | {
        id: string;
        fitness: number;
        metadata?: Record<string, any> | undefined;
        generation?: number | undefined;
        genes?: any;
        parent_ids?: string[] | undefined;
    } | undefined;
    pes_result?: {
        state: "running" | "completed" | "failed" | "pending" | "cancelled" | "timeout" | "paused";
        metrics: {
            duration_ms: number;
            success: boolean;
            error_count: number;
            steps_completed?: number | undefined;
            steps_failed?: number | undefined;
            score?: number | undefined;
            iterations?: number | undefined;
            memory_peak_mb?: number | undefined;
            cpu_time_ms?: number | undefined;
        };
        id: string;
        created_at: string;
        problem_id: string;
        plan_id: string;
        outputs: Record<string, any>;
        correlation_id?: string | undefined;
        error?: {
            message: string;
            code: string;
            details?: Record<string, any> | undefined;
            stack_trace?: string | undefined;
        } | undefined;
        metadata?: Record<string, any> | undefined;
        completed_at?: string | undefined;
        logs?: {
            timestamp: string;
            level: "debug" | "info" | "warn" | "error" | "fatal";
            message: string;
            correlation_id?: string | undefined;
            step_id?: string | undefined;
            context?: Record<string, any> | undefined;
        }[] | undefined;
        artifacts?: {
            type: string;
            uri: string;
            metadata?: Record<string, any> | undefined;
            size_bytes?: number | undefined;
            checksum?: string | undefined;
        }[] | undefined;
    } | undefined;
    evolution_result?: {
        best_fitness: number;
        generations: number;
        population: {
            id: string;
            fitness: number;
            metadata?: Record<string, any> | undefined;
            generation?: number | undefined;
            genes?: any;
            parent_ids?: string[] | undefined;
        }[];
        metadata?: Record<string, any> | undefined;
        converged?: boolean | undefined;
        convergence_generation?: number | undefined;
        diversity_score?: number | undefined;
    } | undefined;
    knowledge_extracted?: {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    }[] | undefined;
}>;
export type HybridExecutionResult = z.infer<typeof HybridExecutionResult>;
/**
 * Adaptive Trigger Schema
 *
 * Represents a trigger for adaptive behavior changes.
 */
export declare const AdaptiveTrigger: z.ZodObject<{
    id: z.ZodString;
    condition: z.ZodEnum<["low_confidence", "high_complexity", "stagnation", "timeout_risk", "convergence_detected", "divergence_detected"]>;
    threshold: z.ZodNumber;
    action: z.ZodEnum<["switch_to_evolution", "switch_to_pes", "increase_iterations", "call_for_help", "merge_populations", "reset_and_restart"]>;
    cooldown_ms: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodOptional<z.ZodString>;
    last_triggered_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    threshold: number;
    action: "switch_to_evolution" | "switch_to_pes" | "increase_iterations" | "call_for_help" | "merge_populations" | "reset_and_restart";
    id: string;
    condition: "low_confidence" | "high_complexity" | "stagnation" | "timeout_risk" | "convergence_detected" | "divergence_detected";
    metadata?: Record<string, any> | undefined;
    created_at?: string | undefined;
    cooldown_ms?: number | undefined;
    last_triggered_at?: string | undefined;
}, {
    threshold: number;
    action: "switch_to_evolution" | "switch_to_pes" | "increase_iterations" | "call_for_help" | "merge_populations" | "reset_and_restart";
    id: string;
    condition: "low_confidence" | "high_complexity" | "stagnation" | "timeout_risk" | "convergence_detected" | "divergence_detected";
    metadata?: Record<string, any> | undefined;
    created_at?: string | undefined;
    cooldown_ms?: number | undefined;
    last_triggered_at?: string | undefined;
}>;
export type AdaptiveTrigger = z.infer<typeof AdaptiveTrigger>;
/**
 * Knowledge Transfer Schema
 *
 * Represents knowledge transferred between paradigms.
 */
export declare const KnowledgeTransfer: z.ZodObject<{
    id: z.ZodString;
    source_paradigm: z.ZodEnum<["pes", "evolution", "hybrid"]>;
    destination_paradigm: z.ZodEnum<["pes", "evolution", "hybrid"]>;
    knowledge: z.ZodObject<{
        id: z.ZodString;
        source_type: z.ZodEnum<["loongflow_solution", "openevolve_optimization", "hybrid_result", "external_solution"]>;
        problem_id: z.ZodString;
        knowledge_type: z.ZodEnum<["solution_pattern", "planning_strategy", "execution_approach", "parameter_setting"]>;
        content: z.ZodObject<{
            pattern: z.ZodString;
            success_rate: z.ZodNumber;
            avg_score: z.ZodNumber;
            usage_count: z.ZodNumber;
            context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        }, {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        }>;
        source_id: z.ZodOptional<z.ZodString>;
        valid_until: z.ZodOptional<z.ZodString>;
        extracted_at: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    }, {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    }>;
    status: z.ZodEnum<["pending", "in_progress", "completed", "failed"]>;
    transfer_result: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    initiated_at: z.ZodString;
    completed_at: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    status: "completed" | "failed" | "pending" | "in_progress";
    id: string;
    knowledge: {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    };
    source_paradigm: "hybrid" | "evolution" | "pes";
    destination_paradigm: "hybrid" | "evolution" | "pes";
    initiated_at: string;
    metadata?: Record<string, any> | undefined;
    completed_at?: string | undefined;
    transfer_result?: Record<string, any> | undefined;
}, {
    status: "completed" | "failed" | "pending" | "in_progress";
    id: string;
    knowledge: {
        id: string;
        content: {
            pattern: string;
            success_rate: number;
            avg_score: number;
            usage_count: number;
            context?: Record<string, any> | undefined;
        };
        source_type: "loongflow_solution" | "openevolve_optimization" | "hybrid_result" | "external_solution";
        problem_id: string;
        knowledge_type: "solution_pattern" | "planning_strategy" | "execution_approach" | "parameter_setting";
        extracted_at: string;
        metadata?: Record<string, any> | undefined;
        source_id?: string | undefined;
        valid_until?: string | undefined;
    };
    source_paradigm: "hybrid" | "evolution" | "pes";
    destination_paradigm: "hybrid" | "evolution" | "pes";
    initiated_at: string;
    metadata?: Record<string, any> | undefined;
    completed_at?: string | undefined;
    transfer_result?: Record<string, any> | undefined;
}>;
export type KnowledgeTransfer = z.infer<typeof KnowledgeTransfer>;
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform LoongFlow solution to evolutionary knowledge
 */
export declare function transformLoongFlowSolutionToKnowledge(solution: LoongFlowSolutionType, problemId: string, knowledgeType: KnowledgeType): EvolutionaryKnowledge;
/**
 * Transform hybrid execution result to summary
 */
export declare function transformHybridResultToSummary(result: HybridExecutionResult): SummaryType;
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate a HybridTask object
 */
export declare function validateHybridTask(data: unknown): {
    success: boolean;
    data?: HybridTask;
    errors?: string[];
};
/**
 * Validate an EvolutionaryKnowledge object
 */
export declare function validateEvolutionaryKnowledge(data: unknown): {
    success: boolean;
    data?: EvolutionaryKnowledge;
    errors?: string[];
};
/**
 * Validate a HybridExecutionResult object
 */
export declare function validateHybridExecutionResult(data: unknown): {
    success: boolean;
    data?: HybridExecutionResult;
    errors?: string[];
};
/**
 * Validate an AdaptiveTrigger object
 */
export declare function validateAdaptiveTrigger(data: unknown): {
    success: boolean;
    data?: AdaptiveTrigger;
    errors?: string[];
};
/**
 * ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================
 */
/**
 * Check if data is a valid HybridTask
 */
export declare function isHybridTask(data: unknown): data is HybridTask;
/**
 * Check if data is a valid HybridExecutionResult
 */
export declare function isHybridExecutionResult(data: unknown): data is HybridExecutionResult;
/**
 * Check if data is a valid EvolutionaryKnowledge
 */
export declare function isEvolutionaryKnowledge(data: unknown): data is EvolutionaryKnowledge;
/**
 * Check if data is a valid AdaptiveTrigger
 */
export declare function isAdaptiveTrigger(data: unknown): data is AdaptiveTrigger;
/**
 * ============================================================================
 * EXAMPLES
 * ============================================================================

/**
 * Example usage:
 * @example
 * ```typescript
 * import {
 *   validateHybridTask,
 *   createPESUTCTimestamp,
 *   createPESCorrelationId
 * } from './hybrid-pes-evolution-canonical';
 *
 * const hybridTask = {
 *   id: createPESCorrelationId(),
 *   type: 'adaptive_execute',
 *   problem: {
 *     id: createPESCorrelationId(),
 *     type: 'optimization',
 *     description: 'Optimize neural network hyperparameters',
 *     context: { dataset: 'MNIST' },
 *     constraints: ['max_layers <= 10'],
 *     success_criteria: ['accuracy > 0.95'],
 *     created_at: createPESUTCTimestamp(),
 *   },
 *   integration_strategy: 'adaptive',
 *   adaptive_triggers: [
 *     {
 *       condition: 'stagnation',
 *       threshold: 5,
 *       action: 'switch_to_evolution',
 *     },
 *     {
 *       condition: 'low_confidence',
 *       threshold: 0.6,
 *       action: 'increase_iterations',
 *     },
 *   ],
 *   created_at: createPESUTCTimestamp(),
 * };
 *
 * const validation = validateHybridTask(hybridTask);
 * if (!validation.success) {
 *   console.error('Invalid hybrid task:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=hybrid-pes-evolution-canonical.d.ts.map