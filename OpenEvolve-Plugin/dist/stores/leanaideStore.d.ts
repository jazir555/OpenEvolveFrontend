/**
 * LeanAide proof status
 */
export type ProofStatus = 'idle' | 'generating' | 'verifying' | 'completed' | 'failed';
/**
 * LeanAide model configuration
 */
export interface LeanAideModelConfig {
    provider: string;
    model: string;
    api_key: string;
    temperature: number;
}
/**
 * Lean 4 code output
 */
export interface LeanCodeOutput {
    code: string;
    language: string;
    structured_output?: any;
}
/**
 * Verification result
 */
export interface VerificationResult {
    success: boolean;
    output: string;
    errors: string[];
    warnings: string[];
    elapsed_time: number;
}
/**
 * LeanAide state interface
 */
interface LeanAideState {
    theorem: string;
    proofAttempt: string;
    generatedProof: LeanCodeOutput | null;
    verificationResult: VerificationResult | null;
    modelConfig: LeanAideModelConfig;
    selectedModels: {
        leanaide: string;
        text: string;
        image: string;
    };
    apiHost: string;
    apiPort: number;
    status: ProofStatus;
    isLoading: boolean;
    error: string | null;
    activeTab: string;
    benchmarkRunning: boolean;
    benchmarkProgress: number;
    benchmarkResults: any[] | null;
    setTheorem: (theorem: string) => void;
    setProofAttempt: (proof: string) => void;
    setGeneratedProof: (proof: LeanCodeOutput) => void;
    setVerificationResult: (result: VerificationResult) => void;
    setModelConfig: (config: Partial<LeanAideModelConfig>) => void;
    setSelectedModel: (type: 'leanaide' | 'text' | 'image', model: string) => void;
    setApiConfig: (host: string, port: number) => void;
    setStatus: (status: ProofStatus) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    setActiveTab: (tab: string) => void;
    setBenchmarkRunning: (running: boolean) => void;
    setBenchmarkProgress: (progress: number) => void;
    setBenchmarkResults: (results: any[]) => void;
    clearOutputs: () => void;
    reset: () => void;
}
/**
 * LeanAide store
 */
export declare const useLeanAideStore: import('zustand').UseBoundStore<Omit<Omit<import('zustand').StoreApi<LeanAideState>, "setState"> & {
    setState<A extends string | {
        type: string;
    }>(partial: LeanAideState | Partial<LeanAideState> | ((state: LeanAideState) => LeanAideState | Partial<LeanAideState>), replace?: boolean, action?: A): void;
}, "persist"> & {
    persist: {
        setOptions: (options: Partial<import('zustand/middleware').PersistOptions<LeanAideState, {
            modelConfig: LeanAideModelConfig;
            selectedModels: {
                leanaide: string;
                text: string;
                image: string;
            };
            apiHost: string;
            apiPort: number;
        }>>) => void;
        clearStorage: () => void;
        rehydrate: () => Promise<void> | void;
        hasHydrated: () => boolean;
        onHydrate: (fn: (state: LeanAideState) => void) => () => void;
        onFinishHydration: (fn: (state: LeanAideState) => void) => () => void;
        getOptions: () => Partial<import('zustand/middleware').PersistOptions<LeanAideState, {
            modelConfig: LeanAideModelConfig;
            selectedModels: {
                leanaide: string;
                text: string;
                image: string;
            };
            apiHost: string;
            apiPort: number;
        }>>;
    };
}>;
export {};
