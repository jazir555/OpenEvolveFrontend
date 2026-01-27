import { BaseIntegration } from '../base/BaseIntegration';
import { ApiClient } from '../api/client';
import type { MakerInputs, MakerResult, ParameterSchema } from '../types';
export declare class MakerIntegration extends BaseIntegration<MakerInputs, MakerResult> {
    name: string;
    version: string;
    description: string;
    constructor(client: ApiClient);
    execute(inputs: MakerInputs): Promise<MakerResult>;
    getSchema(): ParameterSchema;
    createTool(specification: any): Promise<MakerResult>;
    createWorkflow(specification: any): Promise<MakerResult>;
    executeTool(toolId: string, inputs: any): Promise<MakerResult>;
    listTools(): Promise<any[]>;
    getTool(toolId: string): Promise<any>;
}
//# sourceMappingURL=maker.d.ts.map