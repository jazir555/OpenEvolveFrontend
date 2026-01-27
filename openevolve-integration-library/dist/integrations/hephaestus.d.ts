import { BaseIntegration } from '../base/BaseIntegration';
import { ApiClient } from '../api/client';
import type { HephaestusInputs, HephaestusResult, ParameterSchema } from '../types';
export declare class HephaestusIntegration extends BaseIntegration<HephaestusInputs, HephaestusResult> {
    name: string;
    version: string;
    description: string;
    constructor(client: ApiClient);
    execute(inputs: HephaestusInputs): Promise<HephaestusResult>;
    getSchema(): ParameterSchema;
    delegate(task: string, agentType?: string, constraints?: Record<string, any>): Promise<HephaestusResult>;
    orchestrate(workflow: any): Promise<HephaestusResult>;
    monitor(sessionId: string): Promise<HephaestusResult>;
    cancelTask(taskId: string): Promise<void>;
    getSessionStatus(sessionId: string): Promise<any>;
}
//# sourceMappingURL=hephaestus.d.ts.map