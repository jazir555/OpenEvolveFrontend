import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { errorLogger } from '@/utils';

/**
 * Knowledge artifact
 */
export interface KnowledgeArtifact {
  artifact_id: string;
  id?: string; // Alias for artifact_id for compatibility
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
  // Artifacts
  artifacts: KnowledgeArtifact[];
  selectedArtifact: KnowledgeArtifact | null;

  // Graph
  graphData: KnowledgeGraph | null;

  // Search
  searchQuery: string;
  searchFilters: SearchFilters;
  searchResults: KnowledgeArtifact[];

  // UI state
  isLoading: boolean;
  error: string | null;
  viewMode: 'list' | 'grid' | 'graph';

  // Actions
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
export const useKnowledgeStore = create<KnowledgeState>()(
  devtools(
    (set, get) => ({
      artifacts: [],
      selectedArtifact: null,
      graphData: null,
      searchQuery: '',
      searchFilters: {
        query: '',
        tags: [],
      },
      searchResults: [],
      isLoading: false,
      error: null,
      viewMode: 'list',

      setArtifacts: (artifacts) => {
        try {
          set({ artifacts });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setArtifacts', additionalData: { artifacts } }
          );
        }
      },

      addArtifact: (artifact) => {
        try {
          set((state) => ({
            artifacts: [artifact, ...state.artifacts],
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'addArtifact', additionalData: { artifact } }
          );
        }
      },

      updateArtifact: (id, updates) => {
        try {
          set((state) => ({
            artifacts: state.artifacts.map((a) =>
              a.artifact_id === id ? { ...a, ...updates } : a
            ),
            selectedArtifact: state.selectedArtifact?.artifact_id === id
              ? { ...state.selectedArtifact, ...updates }
              : state.selectedArtifact,
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'updateArtifact', additionalData: { id, updates } }
          );
        }
      },

      removeArtifact: (id) => {
        try {
          set((state) => ({
            artifacts: state.artifacts.filter((a) => a.artifact_id !== id),
            selectedArtifact: state.selectedArtifact?.artifact_id === id
              ? null
              : state.selectedArtifact,
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'removeArtifact', additionalData: { id } }
          );
        }
      },

      setSelectedArtifact: (artifact) => {
        try {
          set({ selectedArtifact: artifact });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setSelectedArtifact', additionalData: { artifact } }
          );
        }
      },

      setGraphData: (graph) => {
        try {
          set({ graphData: graph });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setGraphData', additionalData: { graph } }
          );
        }
      },

      setSearchQuery: (query) => {
        try {
          set((state) => ({
            searchQuery: query,
            searchFilters: { ...state.searchFilters, query },
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setSearchQuery', additionalData: { query } }
          );
        }
      },

      setSearchFilters: (filters) => {
        try {
          set((state) => ({
            searchFilters: { ...state.searchFilters, ...filters },
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setSearchFilters', additionalData: { filters } }
          );
        }
      },

      clearSearchFilters: () => {
        try {
          set({
            searchFilters: {
              query: '',
              tags: [],
            },
            searchQuery: '',
          });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'clearSearchFilters' }
          );
        }
      },

      setLoading: (loading) => {
        try {
          set({ isLoading: loading });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setLoading', additionalData: { loading } }
          );
        }
      },

      setError: (error) => {
        try {
          set({ error });
        } catch (setErrorError) {
          errorLogger.logError(
            setErrorError instanceof Error ? setErrorError : new Error(String(setErrorError)),
            'error',
            { component: 'KnowledgeStore', function: 'setError', additionalData: { error } }
          );
        }
      },

      setViewMode: (mode) => {
        try {
          set({ viewMode: mode });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'setViewMode', additionalData: { mode } }
          );
        }
      },

      reset: () => {
        try {
          set({
            selectedArtifact: null,
            searchResults: [],
            error: null,
          });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'KnowledgeStore', function: 'reset' }
          );
        }
      },
    }),
    { name: 'KnowledgeStore' }
  )
);
