import { BaseIntegration } from '../base/BaseIntegration';
import { ApiClient } from '../api/client';
import type { EvolutionInputs, EvolutionResult, ParameterSchema } from '../types';
export declare class EvolutionIntegration extends BaseIntegration<EvolutionInputs, EvolutionResult> {
    name: string;
    version: string;
    description: string;
    constructor(client: ApiClient);
    execute(inputs: EvolutionInputs): Promise<EvolutionResult>;
    getSchema(): ParameterSchema;
    evolve(population: any[], generations: number): Promise<EvolutionResult>;
    evolveAdversarial(baseSolution: any, attackStrategies?: string[]): Promise<EvolutionResult>;
}
//# sourceMappingURL=evolution.d.ts.map