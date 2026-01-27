import { BaseIntegrationAdapter } from './base';
import type { BackendClient } from '../api/backend';
import type { ValidationResult, ParameterSchema, ProgressUpdate, ExecutionOptions } from '../api/types';
import type { LeanAideInputs, LeanAideResult, TranslationResult, ProofResult, VerificationResult, MCTSResult, MathResult, MCTSConfig } from '../types/leanaide';
export declare class LeanAideIntegration extends BaseIntegrationAdapter {
    constructor(client: BackendClient);
    execute<TInputs = LeanAideInputs, TResult = LeanAideResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    validate<TInputs = LeanAideInputs>(inputs: TInputs): Promise<ValidationResult>;
    getSchema(): ParameterSchema;
    protected getEndpoints(): string[];
    translateTheorem(input: string): Promise<TranslationResult>;
    generateProof(theorem: string, strategy: string): Promise<ProofResult>;
    verifyProof(proof: string): Promise<VerificationResult>;
    runMCTS(problem: string, config: MCTSConfig): Promise<MCTSResult>;
    queryMath(question: string): Promise<MathResult>;
    streamGenerateProof(theorem: string, strategy: string, onProgress: (update: ProgressUpdate) => void, options?: ExecutionOptions): Promise<ProofResult>;
}
//# sourceMappingURL=leanaide.new.d.ts.map