/**
 * Knowledge artifact
 */
export interface KnowledgeArtifact {
    artifact_id: string;
    id?: string;
    title: string;
    content: string;
    language?: string;
    tags: string[];
    version: number;
    created_at: string;
    updated_at: string;
    created_by?: string;
}
/**
 * Knowledge graph node
 */
export interface GraphNode {
    id: string;
    label: string;
    type: string;
    data: Record<string, any>;
}
/**
 * Knowledge graph edge
 */
export interface GraphEdge {
    id: string;
    source: string;
    target: string;
    label?: string;
    type: string;
}
/**
 * Knowledge graph data
 */
export interface KnowledgeGraph {
    nodes: GraphNode[];
    edges: GraphEdge[];
}
/**
 * Search filters
 */
export interface SearchFilters {
    query: string;
    tags: string[];
    language?: string;
    dateFrom?: Date;
    dateTo?: Date;
}
/**
 * Knowledge state interface
 */
interface KnowledgeState {
    artifacts: KnowledgeArtifact[];
    selectedArtifact: KnowledgeArtifact | null;
    graphData: KnowledgeGraph | null;
    searchQuery: string;
    searchFilters: SearchFilters;
    searchResults: KnowledgeArtifact[];
    isLoading: boolean;
    error: string | null;
    viewMode: 'list' | 'grid' | 'graph';
    setArtifacts: (artifacts: KnowledgeArtifact[]) => void;
    addArtifact: (artifact: KnowledgeArtifact) => void;
    updateArtifact: (id: string, updates: Partial<KnowledgeArtifact>) => void;
    removeArtifact: (id: string) => void;
    setSelectedArtifact: (artifact: KnowledgeArtifact | null) => void;
    setGraphData: (graph: KnowledgeGraph) => void;
    setSearchQuery: (query: string) => void;
    setSearchFilters: (filters: Partial<SearchFilters>) => void;
    clearSearchFilters: () => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    setViewMode: (mode: 'list' | 'grid' | 'graph') => void;
    reset: () => void;
}
/**
 * Analytics store
 */
export declare const useKnowledgeStore: import('zustand').UseBoundStore<Omit<import('zustand').StoreApi<KnowledgeState>, "setState"> & {
    setState<A extends string | {
        type: string;
    }>(partial: KnowledgeState | Partial<KnowledgeState> | ((state: KnowledgeState) => KnowledgeState | Partial<KnowledgeState>), replace?: boolean, action?: A): void;
}>;
export {};
