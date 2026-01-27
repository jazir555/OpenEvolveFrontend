import { BaseIntegration } from '../base/BaseIntegration';
import { ApiClient } from '../api/client';
import type { DecompositionInputs, DecompositionResult, ParameterSchema, ValidationResult } from '../types';
export declare class DecompositionIntegration extends BaseIntegration<DecompositionInputs, DecompositionResult> {
    name: string;
    version: string;
    description: string;
    constructor(client: ApiClient);
    execute(inputs: DecompositionInputs): Promise<DecompositionResult>;
    getSchema(): ParameterSchema;
    analyzeDependencies(subproblems: any[]): Promise<any>;
    validateDecomposition(result: DecompositionResult): Promise<ValidationResult>;
}
//# sourceMappingURL=decomposition.d.ts.map