export interface SearchResult {
    id: string;
    title: string;
    snippet: string;
    type: string;
    score: number;
}
interface KnowledgeSearchProps {
    onSearch: (query: string, filters?: SearchFilters) => Promise<SearchResult[]>;
    onResultClick?: (result: SearchResult) => void;
    className?: string;
}
export interface SearchFilters {
    types?: string[];
    dateRange?: {
        start: Date;
        end: Date;
    };
}
export declare function KnowledgeSearch({ onSearch, onResultClick, className }: KnowledgeSearchProps): import("react/jsx-runtime").JSX.Element;
export {};
