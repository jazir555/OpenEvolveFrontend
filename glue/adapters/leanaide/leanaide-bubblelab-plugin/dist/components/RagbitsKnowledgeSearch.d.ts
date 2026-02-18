import { RagbitsSearchResult } from '../lib/ragbitsClient';
export interface RagbitsKnowledgeSearchProps {
    initialQuery?: string;
    filters?: Record<string, unknown>;
    topK?: number;
    onResults?: (results: RagbitsSearchResult[]) => void;
    className?: string;
}
export declare function RagbitsKnowledgeSearch({ initialQuery, filters, topK, onResults, className, }: RagbitsKnowledgeSearchProps): any;
