import { LeanCodeOutput, VerificationResult, ProofStatus } from '../stores/leanaideStore';
/**
 * Lean 4 proof generation parameters
 */
export interface LeanProofParams {
    theorem: string;
    proof_attempt?: string;
    model: string;
    temperature: number;
}
/**
 * Proof verification params
 */
export interface VerificationParams {
    code: string;
    timeout?: number;
}
/**
 * LeanAIDE state
 */
export interface LeanAIDEState {
    data: {
        generatedProof: LeanCodeOutput | null;
        verificationResult: VerificationResult | null;
    } | null;
    loading: boolean;
    error: Error | null;
    progress: number;
    status: ProofStatus;
}
/**
 * Custom hook for Lean 4 formal verification
 * Manages theorem proving and proof verification workflows
 */
export declare function useLeanAIDE(): {
    theorem: string;
    proofAttempt: string;
    modelConfig: import('../stores/leanaideStore').LeanAideModelConfig;
    execute: (params: LeanProofParams) => Promise<LeanCodeOutput | null>;
    verify: (params: VerificationParams) => Promise<VerificationResult | null>;
    getModels: () => Promise<Array<{
        provider: string;
        models: string[];
    }>>;
    getStatus: () => ProofStatus;
    getResults: () => {
        generatedProof: LeanCodeOutput | null;
        verificationResult: VerificationResult | null;
    };
    cancel: () => void;
    reset: () => void;
    updateModelConfig: (config: Partial<import('../stores/leanaideStore').LeanAideModelConfig>) => void;
    runBenchmark: (dataset: any[], model: string, evaluator: string) => Promise<string | null>;
    getBenchmarkResults: (benchmarkId: string) => Promise<any[] | null>;
    data: {
        generatedProof: LeanCodeOutput | null;
        verificationResult: VerificationResult | null;
    } | null;
    loading: boolean;
    error: Error | null;
    progress: number;
    status: ProofStatus;
};
/**
 * Lean 4 tactics library hook
 */
export declare function useLeanTactics(): {
    refetch: (category?: string) => Promise<void>;
    data: Array<{
        name: string;
        description: string;
        syntax: string;
        example: string;
        category: string;
    }> | null;
    loading: boolean;
    error: Error | null;
};
/**
 * Lean 4 documentation hook
 */
export declare function useLeanDocs(): {
    refetch: () => Promise<void>;
    data: {
        library_docs: string;
        tactic_reference: string;
        examples: string;
    } | null;
    loading: boolean;
    error: Error | null;
};
