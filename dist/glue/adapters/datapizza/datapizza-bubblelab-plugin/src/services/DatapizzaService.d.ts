import { DatapizzaClient } from './DatapizzaClient';
import { DatapizzaPipelineResult, DatapizzaProcessingResult, DatapizzaQueryResult } from '../types/plugin-types';
export declare class DatapizzaService {
    private client;
    constructor(client: DatapizzaClient);
    runPipeline(dataSource: string, pipelineType: string): Promise<DatapizzaPipelineResult>;
    processData(data: any, processingType?: string): Promise<DatapizzaProcessingResult>;
    queryData(query: string, dataSource?: string): Promise<DatapizzaQueryResult>;
    getPipelineRecommendation(dataSource: string, context?: string): Promise<string>;
    detectDataDomain(data: any): Promise<string | null>;
    isProcessableData(data: any): Promise<boolean>;
    clearCache(): Promise<void>;
}
//# sourceMappingURL=DatapizzaService.d.ts.map