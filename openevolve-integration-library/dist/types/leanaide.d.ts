import { ExecutionConfig } from './common';
export interface LeanAideInputs {
    operation: 'translate' | 'prove' | 'verify' | 'mcts' | 'query';
    input: string | LeanAideProofInput | LeanAideMCTSInput | LeanAideQueryInput;
    config?: ExecutionConfig;
}
export interface LeanAideTranslationInput {
    theorem: string;
    from?: 'natural' | 'formal';
    target?: 'lean4' | 'isabelle' | 'coq';
}
export interface LeanAideProofInput {
    theorem: string;
    strategy: string;
    tactics?: string[];
    context?: string[];
    timeout?: number;
}
export interface LeanAideMCTSInput {
    problem: string;
    config: MCTSConfig;
}
export interface MCTSConfig {
    simulations: number;
    explorationConstant?: number;
    maxDepth?: number;
    rolloutStrategy?: 'random' | 'guided' | 'heuristic';
    selectionStrategy?: 'uct' | 'thompson' | 'epsilon-greedy';
}
export interface LeanAideQueryInput {
    question: string;
    domain?: 'algebra' | 'analysis' | 'geometry' | 'topology' | 'logic';
    detail?: 'brief' | 'standard' | 'detailed';
}
export interface TranslationResult {
    original: string;
    translated: string;
    confidence: number;
    notes?: string[];
    verified?: boolean;
}
export interface ProofResult {
    proof: string;
    steps: ProofStep[];
    tactics: string[];
    status: 'proven' | 'partial' | 'failed';
    confidence: number;
    executionTime: number;
    proofTree?: any;
}
export interface ProofStep {
    step: number;
    tactic: string;
    goalBefore: string;
    goalAfter: string;
}
export interface VerificationResult {
    valid: boolean;
    message: string;
    errors?: VerificationError[];
    warnings?: string[];
    verificationTime: number;
}
export interface VerificationError {
    location: string;
    message: string;
    severity: 'error' | 'warning';
}
export interface MCTSResult {
    bestSolution: string;
    score: number;
    simulations: number;
    statistics: MCTSStatistics;
}
export interface MCTSStatistics {
    nodesVisited: number;
    nodesExpanded: number;
    averageDepth: number;
    maxDepth: number;
}
export interface MathResult {
    answer: string;
    explanation: string;
    references?: string[];
}
export interface LeanAideResult {
    type: 'translation' | 'proof' | 'verification' | 'mcts' | 'query';
    result: TranslationResult | ProofResult | VerificationResult | MCTSResult | MathResult;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=leanaide.d.ts.map