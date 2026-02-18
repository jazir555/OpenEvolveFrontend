import { DatapizzaQueryResult } from '../types/plugin-types';
import { DatapizzaClient } from '../services/DatapizzaClient';
interface DatapizzaQueryOptions {
    dataSource?: string;
    maxResults?: number;
    threshold?: number;
    includeMetadata?: boolean;
}
export declare function useDatapizzaQuery(client?: DatapizzaClient): {
    queryData: (query: string, options?: DatapizzaQueryOptions) => Promise<DatapizzaQueryResult>;
    isLoading: boolean;
    error: string | null;
};
export {};
