import { BaseIntegration } from '../base/BaseIntegration';
import { ApiClient } from '../api/client';
import type { KnowledgeInputs, KnowledgeResult, ParameterSchema } from '../types';
export declare class KnowledgeIntegration extends BaseIntegration<KnowledgeInputs, KnowledgeResult> {
    name: string;
    version: string;
    description: string;
    constructor(client: ApiClient);
    execute(inputs: KnowledgeInputs): Promise<KnowledgeResult>;
    getSchema(): ParameterSchema;
    extract(source: any, extractionType?: string): Promise<KnowledgeResult>;
    queryGraph(graphId: string, query: string): Promise<KnowledgeResult>;
    updateGraph(graphId: string, additions: any[], deletions: any[]): Promise<KnowledgeResult>;
}
//# sourceMappingURL=knowledge.d.ts.map