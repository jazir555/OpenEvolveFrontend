/**
 * Code generation parameters
 */
export interface CodeGenerationParams {
    requirement: string;
    language: string;
    framework?: string;
    context?: string;
    include_tests?: boolean;
    include_comments?: boolean;
    style_guide?: string;
}
/**
 * Generated code result
 */
export interface GeneratedCode {
    code_id: string;
    language: string;
    code: string;
    tests?: string;
    documentation?: string;
    metadata: {
        generated_at: string;
        model_used: string;
        confidence_score: number;
        tokens_used: number;
    };
}
/**
 * Code review result
 */
export interface CodeReviewResult {
    review_id: string;
    code_id: string;
    issues: Array<{
        severity: 'low' | 'medium' | 'high' | 'critical';
        category: string;
        description: string;
        suggestion?: string;
        line_number?: number;
    }>;
    metrics: {
        complexity: number;
        maintainability: number;
        test_coverage?: number;
        documentation_coverage: number;
    };
    overall_score: number;
}
/**
 * Code optimization result
 */
export interface CodeOptimizationResult {
    optimized_code: string;
    improvements: Array<{
        type: string;
        description: string;
        impact: 'low' | 'medium' | 'high';
    }>;
    performance_gain: number;
}
/**
 * Hephaestus state
 */
export interface HephaestusState {
    data: GeneratedCode | CodeReviewResult | CodeOptimizationResult | null;
    loading: boolean;
    error: Error | null;
    progress: number;
}
/**
 * Custom hook for Hephaestus code generation bridge
 * Manages code generation, review, and optimization workflows
 */
export declare function useHephaestus(codeId?: string): {
    currentOperation: "generate" | "optimize" | "review";
    execute: (params: CodeGenerationParams) => Promise<GeneratedCode | null>;
    review: (code: string, language: string) => Promise<CodeReviewResult | null>;
    optimize: (code: string, language: string, optimization_goals?: string[]) => Promise<CodeOptimizationResult | null>;
    getStatus: () => Promise<any>;
    getResults: () => (GeneratedCode | CodeReviewResult | CodeOptimizationResult | null);
    cancel: () => void;
    reset: () => void;
    getSupportedLanguages: () => Promise<string[]>;
    getTemplates: (language: string) => Promise<Array<{
        name: string;
        description: string;
        template: string;
    }>>;
    applyFix: (code: string, fixIndex: number) => Promise<string | null>;
    getCodeMetrics: (code: string, language: string) => Promise<{
        lines_of_code: number;
        complexity: number;
        maintainability_index: number;
        technical_debt: number;
    } | null>;
    data: GeneratedCode | CodeReviewResult | CodeOptimizationResult | null;
    loading: boolean;
    error: Error | null;
    progress: number;
};
/**
 * Hephaestus templates hook
 */
export declare function useHephaestusTemplates(): {
    refetch: (language?: string, category?: string) => Promise<void>;
    data: Array<{
        id: string;
        name: string;
        language: string;
        category: string;
        description: string;
        template: string;
    }>;
    loading: boolean;
    error: Error | null;
};
/**
 * Hephaestus code history hook
 */
export declare function useHephaestusHistory(): {
    refetch: (limit?: number) => Promise<void>;
    data: Array<{
        code_id: string;
        requirement: string;
        language: string;
        generated_at: string;
    }>;
    loading: boolean;
    error: Error | null;
};
