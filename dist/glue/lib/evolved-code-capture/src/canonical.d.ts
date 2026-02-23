/**
 * Evolved Code Capture - Canonical Schema Definitions
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: These are canonical schemas, no core project imports
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Configuration Explicitness: No magic defaults
 * - Anti-Corruption Layer: Normalize all evolved code to this schema
 *
 * This schema defines the canonical data models for capturing evolved code
 * from OpenEvolve and storing it in knowledge systems (Vector DB + Graphiti).
 */
import { z } from 'zod';
/**
 * Problem Type Classification
 */
export declare const ProblemTypeEnum: z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>;
export type ProblemType = z.infer<typeof ProblemTypeEnum>;
/**
 * Constraints definition
 */
export declare const ConstraintsSchema: z.ZodObject<{
    max_memory_mb: z.ZodOptional<z.ZodNumber>;
    max_runtime_ms: z.ZodOptional<z.ZodNumber>;
    required_libraries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    forbidden_patterns: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    language_version: z.ZodOptional<z.ZodString>;
    target_platform: z.ZodOptional<z.ZodString>;
    custom_constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    custom_constraints?: Record<string, any> | undefined;
    forbidden_patterns?: string[] | undefined;
    max_memory_mb?: number | undefined;
    max_runtime_ms?: number | undefined;
    required_libraries?: string[] | undefined;
    language_version?: string | undefined;
    target_platform?: string | undefined;
}, {
    custom_constraints?: Record<string, any> | undefined;
    forbidden_patterns?: string[] | undefined;
    max_memory_mb?: number | undefined;
    max_runtime_ms?: number | undefined;
    required_libraries?: string[] | undefined;
    language_version?: string | undefined;
    target_platform?: string | undefined;
}>;
export type Constraints = z.infer<typeof ConstraintsSchema>;
/**
 * Problem definition
 */
export declare const ProblemSchema: z.ZodObject<{
    description: z.ZodString;
    type: z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>;
    constraints: z.ZodOptional<z.ZodObject<{
        max_memory_mb: z.ZodOptional<z.ZodNumber>;
        max_runtime_ms: z.ZodOptional<z.ZodNumber>;
        required_libraries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        forbidden_patterns: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        language_version: z.ZodOptional<z.ZodString>;
        target_platform: z.ZodOptional<z.ZodString>;
        custom_constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        custom_constraints?: Record<string, any> | undefined;
        forbidden_patterns?: string[] | undefined;
        max_memory_mb?: number | undefined;
        max_runtime_ms?: number | undefined;
        required_libraries?: string[] | undefined;
        language_version?: string | undefined;
        target_platform?: string | undefined;
    }, {
        custom_constraints?: Record<string, any> | undefined;
        forbidden_patterns?: string[] | undefined;
        max_memory_mb?: number | undefined;
        max_runtime_ms?: number | undefined;
        required_libraries?: string[] | undefined;
        language_version?: string | undefined;
        target_platform?: string | undefined;
    }>>;
    input_spec: z.ZodOptional<z.ZodString>;
    output_spec: z.ZodOptional<z.ZodString>;
    test_cases: z.ZodOptional<z.ZodArray<z.ZodObject<{
        input: z.ZodAny;
        expected_output: z.ZodAny;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        input?: any;
        description?: string | undefined;
        expected_output?: any;
    }, {
        input?: any;
        description?: string | undefined;
        expected_output?: any;
    }>, "many">>;
    difficulty: z.ZodOptional<z.ZodEnum<["easy", "medium", "hard", "expert"]>>;
    tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
    description: string;
    metadata?: Record<string, any> | undefined;
    constraints?: {
        custom_constraints?: Record<string, any> | undefined;
        forbidden_patterns?: string[] | undefined;
        max_memory_mb?: number | undefined;
        max_runtime_ms?: number | undefined;
        required_libraries?: string[] | undefined;
        language_version?: string | undefined;
        target_platform?: string | undefined;
    } | undefined;
    tags?: string[] | undefined;
    difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
    input_spec?: string | undefined;
    output_spec?: string | undefined;
    test_cases?: {
        input?: any;
        description?: string | undefined;
        expected_output?: any;
    }[] | undefined;
}, {
    type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
    description: string;
    metadata?: Record<string, any> | undefined;
    constraints?: {
        custom_constraints?: Record<string, any> | undefined;
        forbidden_patterns?: string[] | undefined;
        max_memory_mb?: number | undefined;
        max_runtime_ms?: number | undefined;
        required_libraries?: string[] | undefined;
        language_version?: string | undefined;
        target_platform?: string | undefined;
    } | undefined;
    tags?: string[] | undefined;
    difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
    input_spec?: string | undefined;
    output_spec?: string | undefined;
    test_cases?: {
        input?: any;
        description?: string | undefined;
        expected_output?: any;
    }[] | undefined;
}>;
export type Problem = z.infer<typeof ProblemSchema>;
/**
 * Metrics from the evolution process
 */
export declare const EvolutionMetricsSchema: z.ZodObject<{
    iterations: z.ZodNumber;
    fitness_score: z.ZodNumber;
    fitness_improvement: z.ZodNumber;
    duration_ms: z.ZodNumber;
    generations: z.ZodOptional<z.ZodNumber>;
    population_size: z.ZodOptional<z.ZodNumber>;
    mutation_rate: z.ZodOptional<z.ZodNumber>;
    crossover_rate: z.ZodOptional<z.ZodNumber>;
    convergence_generation: z.ZodOptional<z.ZodNumber>;
    total_evaluations: z.ZodOptional<z.ZodNumber>;
    success_rate: z.ZodOptional<z.ZodNumber>;
    benchmark_score: z.ZodOptional<z.ZodNumber>;
    additional_metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    duration_ms: number;
    iterations: number;
    fitness_score: number;
    fitness_improvement: number;
    success_rate?: number | undefined;
    population_size?: number | undefined;
    generations?: number | undefined;
    mutation_rate?: number | undefined;
    crossover_rate?: number | undefined;
    convergence_generation?: number | undefined;
    total_evaluations?: number | undefined;
    benchmark_score?: number | undefined;
    additional_metrics?: Record<string, any> | undefined;
}, {
    duration_ms: number;
    iterations: number;
    fitness_score: number;
    fitness_improvement: number;
    success_rate?: number | undefined;
    population_size?: number | undefined;
    generations?: number | undefined;
    mutation_rate?: number | undefined;
    crossover_rate?: number | undefined;
    convergence_generation?: number | undefined;
    total_evaluations?: number | undefined;
    benchmark_score?: number | undefined;
    additional_metrics?: Record<string, any> | undefined;
}>;
export type EvolutionMetrics = z.infer<typeof EvolutionMetricsSchema>;
/**
 * Programming language
 */
export declare const LanguageEnum: z.ZodEnum<["python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go", "rust", "julia", "matlab", "r", "other"]>;
export type Language = z.infer<typeof LanguageEnum>;
/**
 * The evolved solution code
 */
export declare const EvolvedCodeSchema: z.ZodObject<{
    id: z.ZodString;
    problem: z.ZodObject<{
        description: z.ZodString;
        type: z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>;
        constraints: z.ZodOptional<z.ZodObject<{
            max_memory_mb: z.ZodOptional<z.ZodNumber>;
            max_runtime_ms: z.ZodOptional<z.ZodNumber>;
            required_libraries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            forbidden_patterns: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            language_version: z.ZodOptional<z.ZodString>;
            target_platform: z.ZodOptional<z.ZodString>;
            custom_constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        }, {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        }>>;
        input_spec: z.ZodOptional<z.ZodString>;
        output_spec: z.ZodOptional<z.ZodString>;
        test_cases: z.ZodOptional<z.ZodArray<z.ZodObject<{
            input: z.ZodAny;
            expected_output: z.ZodAny;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }, {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }>, "many">>;
        difficulty: z.ZodOptional<z.ZodEnum<["easy", "medium", "hard", "expert"]>>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    }, {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    }>;
    language: z.ZodEnum<["python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go", "rust", "julia", "matlab", "r", "other"]>;
    code: z.ZodString;
    function_name: z.ZodOptional<z.ZodString>;
    class_name: z.ZodOptional<z.ZodString>;
    metrics: z.ZodObject<{
        iterations: z.ZodNumber;
        fitness_score: z.ZodNumber;
        fitness_improvement: z.ZodNumber;
        duration_ms: z.ZodNumber;
        generations: z.ZodOptional<z.ZodNumber>;
        population_size: z.ZodOptional<z.ZodNumber>;
        mutation_rate: z.ZodOptional<z.ZodNumber>;
        crossover_rate: z.ZodOptional<z.ZodNumber>;
        convergence_generation: z.ZodOptional<z.ZodNumber>;
        total_evaluations: z.ZodOptional<z.ZodNumber>;
        success_rate: z.ZodOptional<z.ZodNumber>;
        benchmark_score: z.ZodOptional<z.ZodNumber>;
        additional_metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        duration_ms: number;
        iterations: number;
        fitness_score: number;
        fitness_improvement: number;
        success_rate?: number | undefined;
        population_size?: number | undefined;
        generations?: number | undefined;
        mutation_rate?: number | undefined;
        crossover_rate?: number | undefined;
        convergence_generation?: number | undefined;
        total_evaluations?: number | undefined;
        benchmark_score?: number | undefined;
        additional_metrics?: Record<string, any> | undefined;
    }, {
        duration_ms: number;
        iterations: number;
        fitness_score: number;
        fitness_improvement: number;
        success_rate?: number | undefined;
        population_size?: number | undefined;
        generations?: number | undefined;
        mutation_rate?: number | undefined;
        crossover_rate?: number | undefined;
        convergence_generation?: number | undefined;
        total_evaluations?: number | undefined;
        benchmark_score?: number | undefined;
        additional_metrics?: Record<string, any> | undefined;
    }>;
    timestamp_utc: z.ZodString;
    is_valid: z.ZodBoolean;
    validation_errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    execution_time_ms: z.ZodOptional<z.ZodNumber>;
    memory_used_mb: z.ZodOptional<z.ZodNumber>;
    parent_code_id: z.ZodOptional<z.ZodString>;
    generation_number: z.ZodOptional<z.ZodNumber>;
    tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    metrics: {
        duration_ms: number;
        iterations: number;
        fitness_score: number;
        fitness_improvement: number;
        success_rate?: number | undefined;
        population_size?: number | undefined;
        generations?: number | undefined;
        mutation_rate?: number | undefined;
        crossover_rate?: number | undefined;
        convergence_generation?: number | undefined;
        total_evaluations?: number | undefined;
        benchmark_score?: number | undefined;
        additional_metrics?: Record<string, any> | undefined;
    };
    id: string;
    problem: {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    };
    code: string;
    timestamp_utc: string;
    language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
    is_valid: boolean;
    metadata?: Record<string, any> | undefined;
    validation_errors?: string[] | undefined;
    tags?: string[] | undefined;
    execution_time_ms?: number | undefined;
    memory_used_mb?: number | undefined;
    function_name?: string | undefined;
    class_name?: string | undefined;
    parent_code_id?: string | undefined;
    generation_number?: number | undefined;
}, {
    metrics: {
        duration_ms: number;
        iterations: number;
        fitness_score: number;
        fitness_improvement: number;
        success_rate?: number | undefined;
        population_size?: number | undefined;
        generations?: number | undefined;
        mutation_rate?: number | undefined;
        crossover_rate?: number | undefined;
        convergence_generation?: number | undefined;
        total_evaluations?: number | undefined;
        benchmark_score?: number | undefined;
        additional_metrics?: Record<string, any> | undefined;
    };
    id: string;
    problem: {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    };
    code: string;
    timestamp_utc: string;
    language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
    is_valid: boolean;
    metadata?: Record<string, any> | undefined;
    validation_errors?: string[] | undefined;
    tags?: string[] | undefined;
    execution_time_ms?: number | undefined;
    memory_used_mb?: number | undefined;
    function_name?: string | undefined;
    class_name?: string | undefined;
    parent_code_id?: string | undefined;
    generation_number?: number | undefined;
}>;
export type EvolvedCode = z.infer<typeof EvolvedCodeSchema>;
/**
 * A similar solution found in the knowledge base
 */
export declare const SimilarSolutionSchema: z.ZodObject<{
    evolved_code: z.ZodObject<{
        id: z.ZodString;
        problem: z.ZodObject<{
            description: z.ZodString;
            type: z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>;
            constraints: z.ZodOptional<z.ZodObject<{
                max_memory_mb: z.ZodOptional<z.ZodNumber>;
                max_runtime_ms: z.ZodOptional<z.ZodNumber>;
                required_libraries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                forbidden_patterns: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                language_version: z.ZodOptional<z.ZodString>;
                target_platform: z.ZodOptional<z.ZodString>;
                custom_constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            }, "strip", z.ZodTypeAny, {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            }, {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            }>>;
            input_spec: z.ZodOptional<z.ZodString>;
            output_spec: z.ZodOptional<z.ZodString>;
            test_cases: z.ZodOptional<z.ZodArray<z.ZodObject<{
                input: z.ZodAny;
                expected_output: z.ZodAny;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }, {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }>, "many">>;
            difficulty: z.ZodOptional<z.ZodEnum<["easy", "medium", "hard", "expert"]>>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        }, {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        }>;
        language: z.ZodEnum<["python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go", "rust", "julia", "matlab", "r", "other"]>;
        code: z.ZodString;
        function_name: z.ZodOptional<z.ZodString>;
        class_name: z.ZodOptional<z.ZodString>;
        metrics: z.ZodObject<{
            iterations: z.ZodNumber;
            fitness_score: z.ZodNumber;
            fitness_improvement: z.ZodNumber;
            duration_ms: z.ZodNumber;
            generations: z.ZodOptional<z.ZodNumber>;
            population_size: z.ZodOptional<z.ZodNumber>;
            mutation_rate: z.ZodOptional<z.ZodNumber>;
            crossover_rate: z.ZodOptional<z.ZodNumber>;
            convergence_generation: z.ZodOptional<z.ZodNumber>;
            total_evaluations: z.ZodOptional<z.ZodNumber>;
            success_rate: z.ZodOptional<z.ZodNumber>;
            benchmark_score: z.ZodOptional<z.ZodNumber>;
            additional_metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        }, {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        }>;
        timestamp_utc: z.ZodString;
        is_valid: z.ZodBoolean;
        validation_errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        parent_code_id: z.ZodOptional<z.ZodString>;
        generation_number: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    }, {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    }>;
    similarity_score: z.ZodNumber;
    similarity_method: z.ZodEnum<["semantic", "structural", "hybrid"]>;
    problem_similarity: z.ZodOptional<z.ZodNumber>;
    code_similarity: z.ZodOptional<z.ZodNumber>;
    matched_features: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    distance: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    evolved_code: {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    };
    similarity_score: number;
    similarity_method: "semantic" | "hybrid" | "structural";
    problem_similarity?: number | undefined;
    code_similarity?: number | undefined;
    matched_features?: string[] | undefined;
    distance?: number | undefined;
}, {
    evolved_code: {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    };
    similarity_score: number;
    similarity_method: "semantic" | "hybrid" | "structural";
    problem_similarity?: number | undefined;
    code_similarity?: number | undefined;
    matched_features?: string[] | undefined;
    distance?: number | undefined;
}>;
export type SimilarSolution = z.infer<typeof SimilarSolutionSchema>;
/**
 * A node in the evolution tree
 */
export declare const EvolutionNodeSchema: z.ZodObject<{
    code_id: z.ZodString;
    code: z.ZodOptional<z.ZodString>;
    fitness_score: z.ZodNumber;
    timestamp_utc: z.ZodString;
    generation: z.ZodNumber;
    parent_id: z.ZodOptional<z.ZodString>;
    children_ids: z.ZodArray<z.ZodString, "many">;
}, "strip", z.ZodTypeAny, {
    timestamp_utc: string;
    generation: number;
    fitness_score: number;
    code_id: string;
    children_ids: string[];
    code?: string | undefined;
    parent_id?: string | undefined;
}, {
    timestamp_utc: string;
    generation: number;
    fitness_score: number;
    code_id: string;
    children_ids: string[];
    code?: string | undefined;
    parent_id?: string | undefined;
}>;
export type EvolutionNode = z.infer<typeof EvolutionNodeSchema>;
/**
 * Full lineage tracking from initial solution to final evolved solution
 */
export declare const EvolutionLineageSchema: z.ZodObject<{
    root_code_id: z.ZodString;
    final_code_id: z.ZodString;
    total_nodes: z.ZodNumber;
    depth: z.ZodNumber;
    nodes: z.ZodArray<z.ZodObject<{
        code_id: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
        fitness_score: z.ZodNumber;
        timestamp_utc: z.ZodString;
        generation: z.ZodNumber;
        parent_id: z.ZodOptional<z.ZodString>;
        children_ids: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        timestamp_utc: string;
        generation: number;
        fitness_score: number;
        code_id: string;
        children_ids: string[];
        code?: string | undefined;
        parent_id?: string | undefined;
    }, {
        timestamp_utc: string;
        generation: number;
        fitness_score: number;
        code_id: string;
        children_ids: string[];
        code?: string | undefined;
        parent_id?: string | undefined;
    }>, "many">;
    branches: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    depth: number;
    nodes: {
        timestamp_utc: string;
        generation: number;
        fitness_score: number;
        code_id: string;
        children_ids: string[];
        code?: string | undefined;
        parent_id?: string | undefined;
    }[];
    total_nodes: number;
    root_code_id: string;
    final_code_id: string;
    metadata?: Record<string, any> | undefined;
    branches?: number | undefined;
}, {
    depth: number;
    nodes: {
        timestamp_utc: string;
        generation: number;
        fitness_score: number;
        code_id: string;
        children_ids: string[];
        code?: string | undefined;
        parent_id?: string | undefined;
    }[];
    total_nodes: number;
    root_code_id: string;
    final_code_id: string;
    metadata?: Record<string, any> | undefined;
    branches?: number | undefined;
}>;
export type EvolutionLineage = z.infer<typeof EvolutionLineageSchema>;
/**
 * Result of capturing an evolved solution
 */
export declare const CaptureResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    code_id: z.ZodString;
    vector_storage_id: z.ZodOptional<z.ZodString>;
    graph_episode_id: z.ZodOptional<z.ZodString>;
    timestamp_utc: z.ZodString;
    processing_time_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    error: z.ZodOptional<z.ZodString>;
    warnings: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    timestamp_utc: string;
    processing_time_ms: number;
    code_id: string;
    correlation_id?: string | undefined;
    error?: string | undefined;
    warnings?: string[] | undefined;
    vector_storage_id?: string | undefined;
    graph_episode_id?: string | undefined;
}, {
    success: boolean;
    timestamp_utc: string;
    processing_time_ms: number;
    code_id: string;
    correlation_id?: string | undefined;
    error?: string | undefined;
    warnings?: string[] | undefined;
    vector_storage_id?: string | undefined;
    graph_episode_id?: string | undefined;
}>;
export type CaptureResult = z.infer<typeof CaptureResultSchema>;
/**
 * Aggregated metrics for the capture system
 */
export declare const CaptureMetricsSchema: z.ZodObject<{
    total_captures: z.ZodNumber;
    successful_captures: z.ZodNumber;
    failed_captures: z.ZodNumber;
    average_processing_time_ms: z.ZodNumber;
    total_storage_size_bytes: z.ZodOptional<z.ZodNumber>;
    last_capture_timestamp: z.ZodOptional<z.ZodString>;
    problem_type_distribution: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    language_distribution: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
}, "strip", z.ZodTypeAny, {
    total_captures: number;
    successful_captures: number;
    failed_captures: number;
    average_processing_time_ms: number;
    total_storage_size_bytes?: number | undefined;
    last_capture_timestamp?: string | undefined;
    problem_type_distribution?: Record<string, number> | undefined;
    language_distribution?: Record<string, number> | undefined;
}, {
    total_captures: number;
    successful_captures: number;
    failed_captures: number;
    average_processing_time_ms: number;
    total_storage_size_bytes?: number | undefined;
    last_capture_timestamp?: string | undefined;
    problem_type_distribution?: Record<string, number> | undefined;
    language_distribution?: Record<string, number> | undefined;
}>;
export type CaptureMetrics = z.infer<typeof CaptureMetricsSchema>;
/**
 * Request to store code with embedding
 */
export declare const StoreWithEmbeddingRequestSchema: z.ZodObject<{
    evolved_code: z.ZodObject<{
        id: z.ZodString;
        problem: z.ZodObject<{
            description: z.ZodString;
            type: z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>;
            constraints: z.ZodOptional<z.ZodObject<{
                max_memory_mb: z.ZodOptional<z.ZodNumber>;
                max_runtime_ms: z.ZodOptional<z.ZodNumber>;
                required_libraries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                forbidden_patterns: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                language_version: z.ZodOptional<z.ZodString>;
                target_platform: z.ZodOptional<z.ZodString>;
                custom_constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            }, "strip", z.ZodTypeAny, {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            }, {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            }>>;
            input_spec: z.ZodOptional<z.ZodString>;
            output_spec: z.ZodOptional<z.ZodString>;
            test_cases: z.ZodOptional<z.ZodArray<z.ZodObject<{
                input: z.ZodAny;
                expected_output: z.ZodAny;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }, {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }>, "many">>;
            difficulty: z.ZodOptional<z.ZodEnum<["easy", "medium", "hard", "expert"]>>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        }, {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        }>;
        language: z.ZodEnum<["python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go", "rust", "julia", "matlab", "r", "other"]>;
        code: z.ZodString;
        function_name: z.ZodOptional<z.ZodString>;
        class_name: z.ZodOptional<z.ZodString>;
        metrics: z.ZodObject<{
            iterations: z.ZodNumber;
            fitness_score: z.ZodNumber;
            fitness_improvement: z.ZodNumber;
            duration_ms: z.ZodNumber;
            generations: z.ZodOptional<z.ZodNumber>;
            population_size: z.ZodOptional<z.ZodNumber>;
            mutation_rate: z.ZodOptional<z.ZodNumber>;
            crossover_rate: z.ZodOptional<z.ZodNumber>;
            convergence_generation: z.ZodOptional<z.ZodNumber>;
            total_evaluations: z.ZodOptional<z.ZodNumber>;
            success_rate: z.ZodOptional<z.ZodNumber>;
            benchmark_score: z.ZodOptional<z.ZodNumber>;
            additional_metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        }, {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        }>;
        timestamp_utc: z.ZodString;
        is_valid: z.ZodBoolean;
        validation_errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        parent_code_id: z.ZodOptional<z.ZodString>;
        generation_number: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    }, {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    }>;
    embedding: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    evolved_code: {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    };
    correlation_id?: string | undefined;
    embedding?: number[] | undefined;
}, {
    evolved_code: {
        metrics: {
            duration_ms: number;
            iterations: number;
            fitness_score: number;
            fitness_improvement: number;
            success_rate?: number | undefined;
            population_size?: number | undefined;
            generations?: number | undefined;
            mutation_rate?: number | undefined;
            crossover_rate?: number | undefined;
            convergence_generation?: number | undefined;
            total_evaluations?: number | undefined;
            benchmark_score?: number | undefined;
            additional_metrics?: Record<string, any> | undefined;
        };
        id: string;
        problem: {
            type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
            description: string;
            metadata?: Record<string, any> | undefined;
            constraints?: {
                custom_constraints?: Record<string, any> | undefined;
                forbidden_patterns?: string[] | undefined;
                max_memory_mb?: number | undefined;
                max_runtime_ms?: number | undefined;
                required_libraries?: string[] | undefined;
                language_version?: string | undefined;
                target_platform?: string | undefined;
            } | undefined;
            tags?: string[] | undefined;
            difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
            input_spec?: string | undefined;
            output_spec?: string | undefined;
            test_cases?: {
                input?: any;
                description?: string | undefined;
                expected_output?: any;
            }[] | undefined;
        };
        code: string;
        timestamp_utc: string;
        language: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r";
        is_valid: boolean;
        metadata?: Record<string, any> | undefined;
        validation_errors?: string[] | undefined;
        tags?: string[] | undefined;
        execution_time_ms?: number | undefined;
        memory_used_mb?: number | undefined;
        function_name?: string | undefined;
        class_name?: string | undefined;
        parent_code_id?: string | undefined;
        generation_number?: number | undefined;
    };
    correlation_id?: string | undefined;
    embedding?: number[] | undefined;
}>;
export type StoreWithEmbeddingRequest = z.infer<typeof StoreWithEmbeddingRequestSchema>;
/**
 * Request to search for similar solutions
 */
export declare const SearchSimilarRequestSchema: z.ZodObject<{
    problem: z.ZodObject<{
        description: z.ZodString;
        type: z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>;
        constraints: z.ZodOptional<z.ZodObject<{
            max_memory_mb: z.ZodOptional<z.ZodNumber>;
            max_runtime_ms: z.ZodOptional<z.ZodNumber>;
            required_libraries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            forbidden_patterns: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            language_version: z.ZodOptional<z.ZodString>;
            target_platform: z.ZodOptional<z.ZodString>;
            custom_constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        }, {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        }>>;
        input_spec: z.ZodOptional<z.ZodString>;
        output_spec: z.ZodOptional<z.ZodString>;
        test_cases: z.ZodOptional<z.ZodArray<z.ZodObject<{
            input: z.ZodAny;
            expected_output: z.ZodAny;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }, {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }>, "many">>;
        difficulty: z.ZodOptional<z.ZodEnum<["easy", "medium", "hard", "expert"]>>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    }, {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    }>;
    max_results: z.ZodDefault<z.ZodNumber>;
    similarity_threshold: z.ZodDefault<z.ZodNumber>;
    filters: z.ZodOptional<z.ZodObject<{
        language: z.ZodOptional<z.ZodEnum<["python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go", "rust", "julia", "matlab", "r", "other"]>>;
        problem_type: z.ZodOptional<z.ZodEnum<["algorithm_optimization", "code_refactoring", "performance_tuning", "bug_fix", "feature_implementation", "code_migration", "parallelization", "memory_optimization", "numerical_computation", "data_structure_design", "other"]>>;
        min_fitness: z.ZodOptional<z.ZodNumber>;
        date_range: z.ZodOptional<z.ZodObject<{
            start: z.ZodString;
            end: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            start: string;
            end: string;
        }, {
            start: string;
            end: string;
        }>>;
    }, "strip", z.ZodTypeAny, {
        problem_type?: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design" | undefined;
        language?: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r" | undefined;
        date_range?: {
            start: string;
            end: string;
        } | undefined;
        min_fitness?: number | undefined;
    }, {
        problem_type?: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design" | undefined;
        language?: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r" | undefined;
        date_range?: {
            start: string;
            end: string;
        } | undefined;
        min_fitness?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    problem: {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    };
    max_results: number;
    similarity_threshold: number;
    correlation_id?: string | undefined;
    filters?: {
        problem_type?: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design" | undefined;
        language?: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r" | undefined;
        date_range?: {
            start: string;
            end: string;
        } | undefined;
        min_fitness?: number | undefined;
    } | undefined;
}, {
    problem: {
        type: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design";
        description: string;
        metadata?: Record<string, any> | undefined;
        constraints?: {
            custom_constraints?: Record<string, any> | undefined;
            forbidden_patterns?: string[] | undefined;
            max_memory_mb?: number | undefined;
            max_runtime_ms?: number | undefined;
            required_libraries?: string[] | undefined;
            language_version?: string | undefined;
            target_platform?: string | undefined;
        } | undefined;
        tags?: string[] | undefined;
        difficulty?: "medium" | "hard" | "easy" | "expert" | undefined;
        input_spec?: string | undefined;
        output_spec?: string | undefined;
        test_cases?: {
            input?: any;
            description?: string | undefined;
            expected_output?: any;
        }[] | undefined;
    };
    correlation_id?: string | undefined;
    max_results?: number | undefined;
    filters?: {
        problem_type?: "other" | "algorithm_optimization" | "code_refactoring" | "performance_tuning" | "bug_fix" | "feature_implementation" | "code_migration" | "parallelization" | "memory_optimization" | "numerical_computation" | "data_structure_design" | undefined;
        language?: "other" | "python" | "c" | "javascript" | "typescript" | "java" | "cpp" | "csharp" | "go" | "rust" | "julia" | "matlab" | "r" | undefined;
        date_range?: {
            start: string;
            end: string;
        } | undefined;
        min_fitness?: number | undefined;
    } | undefined;
    similarity_threshold?: number | undefined;
}>;
export type SearchSimilarRequest = z.infer<typeof SearchSimilarRequestSchema>;
/**
 * Request to get evolution lineage
 */
export declare const GetLineageRequestSchema: z.ZodObject<{
    code_id: z.ZodString;
    include_code_content: z.ZodDefault<z.ZodBoolean>;
    max_depth: z.ZodOptional<z.ZodNumber>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    code_id: string;
    include_code_content: boolean;
    correlation_id?: string | undefined;
    max_depth?: number | undefined;
}, {
    code_id: string;
    correlation_id?: string | undefined;
    max_depth?: number | undefined;
    include_code_content?: boolean | undefined;
}>;
export type GetLineageRequest = z.infer<typeof GetLineageRequestSchema>;
/**
 * Validate evolved code against canonical schema
 */
export declare function validateEvolvedCode(data: unknown): {
    success: boolean;
    data?: EvolvedCode;
    errors: string[];
};
/**
 * Validate problem against canonical schema
 */
export declare function validateProblem(data: unknown): {
    success: boolean;
    data?: Problem;
    errors: string[];
};
/**
 * Validate evolution metrics against canonical schema
 */
export declare function validateEvolutionMetrics(data: unknown): {
    success: boolean;
    data?: EvolutionMetrics;
    errors: string[];
};
/**
 * Validate capture result against canonical schema
 */
export declare function validateCaptureResult(data: unknown): {
    success: boolean;
    data?: CaptureResult;
    errors: string[];
};
//# sourceMappingURL=canonical.d.ts.map