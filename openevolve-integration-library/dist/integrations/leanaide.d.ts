import { BaseIntegration } from '../base/BaseIntegration';
import { ApiClient } from '../api/client';
import type { LeanAideInputs, LeanAideResult, ParameterSchema } from '../types';
export declare class LeanAideIntegration extends BaseIntegration<LeanAideInputs, LeanAideResult> {
    name: string;
    version: string;
    description: string;
    constructor(client: ApiClient);
    execute(inputs: LeanAideInputs): Promise<LeanAideResult>;
    getSchema(): ParameterSchema;
    verify(lemma: string, tactics: string[]): Promise<any>;
    plan(problem: string, iterations?: number): Promise<any>;
    optimize(problem: string, constraints: Record<string, any>): Promise<any>;
}
//# sourceMappingURL=leanaide.d.ts.map