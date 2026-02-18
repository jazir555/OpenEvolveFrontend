import { DatapizzaPipelineResult } from '../types/plugin-types';
import { DatapizzaClient } from '../services/DatapizzaClient';
interface DatapizzaPipelineOptions {
    pipelineType?: 'standard' | 'advanced' | 'custom';
    dataSource?: string;
    chunkSize?: number;
    overlapSize?: number;
    embeddingModel?: string;
    vectorStore?: string;
    skipValidation?: boolean;
    skipEmbedding?: boolean;
}
export declare function useDatapizzaPipeline(client?: DatapizzaClient): {
    runPipeline: (dataSource: string, options?: DatapizzaPipelineOptions) => Promise<DatapizzaPipelineResult>;
    isRunning: boolean;
    progress: number;
    currentStep: string;
    error: string | null;
};
export {};
