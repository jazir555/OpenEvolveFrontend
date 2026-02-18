import { DatapizzaProcessingResult } from '../types/plugin-types';
import { DatapizzaClient } from '../services/DatapizzaClient';
interface DatapizzaProcessingOptions {
    processingType?: 'standard' | 'advanced' | 'custom';
    chunkSize?: number;
    overlapSize?: number;
    embeddingModel?: string;
    vectorStore?: string;
}
export declare function useDatapizzaProcessing(client?: DatapizzaClient): {
    processData: (data: any, options?: DatapizzaProcessingOptions) => Promise<DatapizzaProcessingResult>;
    isLoading: boolean;
    progress: number;
    error: string | null;
};
export {};
