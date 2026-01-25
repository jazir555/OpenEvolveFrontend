import errorLogger from '@/utils/errorLogging';
// @ts-nocheck
import { useState, useCallback, useEffect, useRef } from 'react';
import { apiClient } from '../services/api/client';
import { knowledgeApi } from '../services/api/endpoints';
import { useKnowledgeStore } from '../stores/knowledgeStore';
import type { KnowledgeArtifact, KnowledgeGraph } from '../stores/knowledgeStore';

/**
 * Knowledge query parameters
 */
export interface KnowledgeQueryParams {
  query: string;
  context?: string;
  limit?: number;
  threshold?: number;
}

/**
 * Knowledge query result
 */
export interface KnowledgeQueryResult {
  artifact_id: string;
  relevance_score: number;
  artifact: KnowledgeArtifact;
  matched_sections: Array<{
    content: string;
    score: number;
  }>;
}

/**
 * Knowledge ingestion parameters
 */
export interface KnowledgeIngestParams {
  content: string;
  title: string;
  language?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

/**
 * Knowledge graph query
 */
export interface GraphQueryParams {
  source?: string;
  relation_type?: string;
  depth?: number;
  max_nodes?: number;
}

/**
 * Knowledge state
 */
export interface KnowledgeEngineState {
  data: any;
  loading: boolean;
  error: Error | null;
  progress: number;
}

/**
 * Custom hook for knowledge engine operations
 * Manages knowledge graph and artifact operations
 */
export function useKnowledgeEngine() {

// Export alias for compatibility
  const [state, setState] = useState<KnowledgeEngineState>({
    data: null,
    loading: false,
    error: null,
    progress: 0,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const {
    artifacts,
    graphData,
    setArtifacts,
    setGraphData,
    addArtifact,
    updateArtifact,
    setLoading,
    setError,
  } = useKnowledgeStore();

  /**
   * Query knowledge base
   */
  const query = useCallback(async (
    params: KnowledgeQueryParams
  ): Promise<KnowledgeQueryResult[]> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);
    setError(null);

    try {
      abortControllerRef.current = new AbortController();

      const response = await knowledgeApi.searchRag({
        query: params.query
      });

      // Map RAG results to KnowledgeQueryResult
      const rawResults = Array.isArray(response)
        ? response
        : Array.isArray(response?.results)
          ? response.results
          : [];
      const results: KnowledgeQueryResult[] = rawResults.map((r: any) => ({
        artifact_id: r.id || 'unknown',
        relevance_score: r.score || 0,
        artifact: {
          artifact_id: r.id || 'unknown',
          title: r.title || 'Untitled',
          content: r.content || '',
          tags: [],
          version: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        matched_sections: [{
          content: r.content || '',
          score: r.score || 0
        }]
      }));

      setState(prev => ({ ...prev, data: results, loading: false }));
      setLoading(false);

      return results;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return [];
    }
  }, [setLoading, setError]);

  /**
   * Ingest knowledge into the graph
   */
  const ingest = useCallback(async (
    params: KnowledgeIngestParams
  ): Promise<KnowledgeArtifact | null> => {
    setState(prev => ({ ...prev, loading: true, error: null, progress: 0 }));
    setLoading(true);
    setError(null);

    let progressInterval: ReturnType<typeof setInterval> | null = null;
    try {
      abortControllerRef.current = new AbortController();
      const safeContent = params.content || '';
      const description = params.metadata?.description || safeContent.slice(0, 200);

      // Simulate progress
      progressInterval = setInterval(() => {
        setState(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 10, 90),
        }));
      }, 200);

      const artifactId = `art-${Date.now()}`;
      const response = await knowledgeApi.add({
        content: safeContent,
        entities: [{
          id: artifactId,
          label: params.title,
          properties: { ...params, created_at: new Date().toISOString() }
        }]
      });

      // Create artifact object from params since API returns success message
      const artifact: KnowledgeArtifact = {
        artifact_id: artifactId,
        id: artifactId,
        title: params.title,
        name: params.title,
        content: safeContent,
        description,
        type: 'Artifact',
        tags: params.tags || [],
        language: params.language,
        version: 1,
        current_version: 1,
        created_at: new Date().toISOString(),
        created: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        updated: new Date().toISOString(),
        created_by: 'System',
        versions: [],
      };

      setState(prev => ({
        ...prev,
        data: artifact,
        loading: false,
        progress: 100,
      }));

      addArtifact(artifact);
      setLoading(false);

      return artifact;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false, progress: 0 }));
      setError(error.message);
      setLoading(false);
      return null;
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    }
  }, [addArtifact, setLoading, setError]);

  /**
   * Get knowledge graph
   */
  const getGraph = useCallback(async (
    params?: GraphQueryParams
  ): Promise<KnowledgeGraph | null> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);
    setError(null);

    try {
      const response = await knowledgeApi.list({ limit: params?.max_nodes });
      const entities = response?.entities || [];
      const relationships = response?.relationships || [];

      const graph: KnowledgeGraph = {
        nodes: entities.map((e: any) => ({
          id: e.id,
          label: e.id,
          type: 'entity',
          data: e.properties || {}
        })),
        edges: relationships.map((r: any) => ({
          id: `${r.source}-${r.target}`,
          source: r.source,
          target: r.target,
          type: r.type || 'rel'
        }))
      };

      setGraphData(graph);
      setState(prev => ({ ...prev, data: graph, loading: false }));
      setLoading(false);

      return graph;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setGraphData, setLoading, setError]);

  /**
   * Get artifacts
   */
  const getArtifacts = useCallback(async (): Promise<KnowledgeArtifact[]> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);
    setError(null);

    try {
      const response = await knowledgeApi.list();
      const entities = response?.entities || [];

      const artifacts: KnowledgeArtifact[] = entities.map((e: any) => ({
        artifact_id: e.id,
        id: e.id,
        title: e.properties?.title || e.id,
        name: e.properties?.title || e.id,
        content: e.properties?.content || '',
        description: e.properties?.description || e.properties?.content?.slice(0, 200) || '',
        type: e.type || 'Artifact',
        tags: e.properties?.tags || [],
        version: e.properties?.version || 1,
        current_version: e.properties?.version || 1,
        created_at: e.properties?.created_at || new Date().toISOString(),
        created: e.properties?.created_at || new Date().toISOString(),
        updated_at: e.properties?.updated_at || new Date().toISOString(),
        updated: e.properties?.updated_at || new Date().toISOString(),
        created_by: e.properties?.created_by || 'System',
        versions: e.properties?.versions || [],
        ...e.properties
      }));

      setArtifacts(artifacts);
      setState(prev => ({ ...prev, data: artifacts, loading: false }));
      setLoading(false);

      return artifacts;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return [];
    }
  }, [setArtifacts, setLoading, setError]);

  /**
   * Get artifact by ID
   */
  const getArtifact = useCallback(async (
    artifactId: string
  ): Promise<KnowledgeArtifact | null> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.getEntity(artifactId);
      if (!response) {
        throw new Error('Artifact not found');
      }
      
      const artifact: KnowledgeArtifact = {
        artifact_id: response.id,
        id: response.id,
        title: response.properties?.title || response.id,
        name: response.properties?.title || response.id,
        content: response.properties?.content || '',
        description: response.properties?.description || response.properties?.content?.slice(0, 200) || '',
        type: response.properties?.type || 'Artifact',
        tags: response.properties?.tags || [],
        version: response.properties?.version || 1,
        current_version: response.properties?.version || 1,
        created_at: response.properties?.created_at || new Date().toISOString(),
        created: response.properties?.created_at || new Date().toISOString(),
        updated_at: response.properties?.updated_at || new Date().toISOString(),
        updated: response.properties?.updated_at || new Date().toISOString(),
        created_by: response.properties?.created_by || 'System',
        versions: response.properties?.versions || [],
        ...response.properties
      };

      setState(prev => ({ ...prev, data: artifact, loading: false }));
      setLoading(false);

      return artifact;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Update artifact
   */
  const updateArtifactData = useCallback(async (
    artifactId: string,
    updates: Partial<KnowledgeArtifact>
  ): Promise<void> => {
    setState(prev => ({ ...prev, loading: true }));
    setLoading(true);

    try {
      await knowledgeApi.updateEntity(artifactId, { properties: updates });

      updateArtifact(artifactId, updates);
      setState(prev => ({ ...prev, loading: false }));
      setLoading(false);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
    }
  }, [updateArtifact, setLoading, setError]);

  /**
   * Delete artifact
   */
  const deleteArtifact = useCallback(async (
    artifactId: string
  ): Promise<void> => {
    setState(prev => ({ ...prev, loading: true }));
    setLoading(true);

    try {
      await knowledgeApi.deleteEntity(artifactId);

      // Remove from store
      const updatedArtifacts = artifacts.filter(a => a.artifact_id !== artifactId);
      setArtifacts(updatedArtifacts);

      setState(prev => ({ ...prev, loading: false }));
      setLoading(false);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
    }
  }, [artifacts, setArtifacts, setLoading, setError]);

  /**
   * Get relationships between artifacts
   */
  const getRelationships = useCallback(async (
    artifactId: string
  ): Promise<Array<{ from: string; to: string; type: string; weight: number }>> => {
    try {
      const response = await apiClient.get<{ relationships: any[] }>(
        `/knowledge/artifacts/${artifactId}/relationships`
      );

      return response?.relationships || [];
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message);
      return [];
    }
  }, [setError]);

  /**
   * Semantic search
   */
  const semanticSearch = useCallback(async (
    query: string,
    limit: number = 10
  ): Promise<KnowledgeQueryResult[]> => {
    setState(prev => ({ ...prev, loading: true }));
    setLoading(true);

    try {
      const response = await knowledgeApi.searchRag({
        query: query
      });

      const rawResults = Array.isArray(response)
        ? response
        : Array.isArray(response?.results)
          ? response.results
          : [];
      const results: KnowledgeQueryResult[] = rawResults.slice(0, limit).map((r: any) => ({
        artifact_id: r.id || 'unknown',
        relevance_score: r.score || 0,
        artifact: {
          artifact_id: r.id || 'unknown',
          title: r.title || 'Untitled',
          content: r.content || '',
          tags: [],
          version: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        matched_sections: []
      }));

      setState(prev => ({ ...prev, data: results, loading: false }));
      setLoading(false);

      return results;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return [];
    }
  }, [setLoading, setError]);

  /**
   * Cancel operation
   */
  const cancel = useCallback((): void => {
    abortControllerRef.current?.abort();
    setState(prev => ({ ...prev, loading: false, progress: 0 }));
    setLoading(false);
  }, [setLoading]);

  /**
   * Reset state
   */
  const reset = useCallback((): void => {
    setState({
      data: null,
      loading: false,
      error: null,
      progress: 0,
    });
    setError(null);
  }, [setError]);

  /**
   * Index a project directory
   */
  const indexProject = useCallback(async (
    params: { projectPath?: string; targetStructure?: string; outputDir?: string }
  ): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.indexProject({
        project_path: params.projectPath,
        target_structure: params.targetStructure,
        output_dir: params.outputDir
      });

      setState(prev => ({ ...prev, data: response, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Ingest a document (PDF, TXT, URL)
   */
  const ingestDocument = useCallback(async (pathOrUrl: string): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.ingestDocument({ path_or_url: pathOrUrl });
      setState(prev => ({ ...prev, data: response, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Generate new knowledge based on context
   */
  const generateKnowledge = useCallback(async (context: string, query: string): Promise<string | null> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.generateKnowledge({ context, query });
      setState(prev => ({ ...prev, data: response, loading: false }));
      setLoading(false);
      return response.generated_knowledge;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Perform unified search across all knowledge types
   */
  const unifiedSearch = useCallback(async (query: string, topK: number = 5): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.unifiedSearch(query, topK);
      setState(prev => ({ ...prev, data: response, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Distill a reusable ACE skill from an artifact
   */
  const distillSkill = useCallback(async (artifactId: string): Promise<boolean> => {
    try {
      const response = await knowledgeApi.distillSkill(artifactId);
      return response.success;
    } catch (err) {
      errorLogger.logError(err, 'error', { component: 'useKnowledgeEngine', function: 'distillSkills', additionalData: {} });
      return false;
    }
  }, []);

  /**
   * Trigger autonomous self-healing
   */
  const selfHeal = useCallback(async (): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.selfHeal();
      setState(prev => ({ ...prev, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Trigger recursive synthesis
   */
  const synthesize = useCallback(async (): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.synthesize();
      setState(prev => ({ ...prev, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Perform deep research on a topic
   */
  const deepResearch = useCallback(async (topic: string): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.deepResearch(topic);
      setState(prev => ({ ...prev, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Formalize and verify a fact
   */
  const verifyFact = useCallback(async (text: string): Promise<any> => {
    try {
      const response = await knowledgeApi.verifyFact(text);
      return response;
    } catch (err) {
      errorLogger.logError(err, 'error', { component: 'useKnowledgeEngine', function: 'verifyFacts', additionalData: {} });
      return null;
    }
  }, []);

  /**
   * Analyze knowledge graph using advanced AI (Karate Club)
   */
  const analyzeGraph = useCallback(async (graphData?: any): Promise<any> => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    setLoading(true);

    try {
      const response = await knowledgeApi.analyze({ graph_data: graphData });
      setState(prev => ({ ...prev, data: response, loading: false }));
      setLoading(false);
      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
      setError(error.message);
      setLoading(false);
      return null;
    }
  }, [setLoading, setError]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return {
    ...state,
    artifacts,
    graphData,
    query,
    ingest,
    getGraph,
    getArtifacts,
    getArtifact,
    updateArtifact: updateArtifactData,
    deleteArtifact,
    getRelationships,
    semanticSearch,
    indexProject,
    cancel,
    reset,
  };
}

/**
 * Knowledge analytics hook
 */
export function useKnowledgeAnalytics() {
  const [state, setState] = useState<{
    data: {
      totalArtifacts: number;
      totalRelationships: number;
      growthRate: number;
      topTags: Array<{ tag: string; count: number }>;
    } | null;
    loading: boolean;
    error: Error | null;
  }>({
    data: null,
    loading: false,
    error: null,
  });

  const fetchAnalytics = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await knowledgeApi.getStatistics();

      setState(prev => ({
        ...prev,
        data: {
          totalArtifacts: response.entity_count,
          totalRelationships: response.relationship_count,
          growthRate: 0, // Not supported by backend yet
          topTags: [] // Not supported by backend yet
        },
        loading: false,
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setState(prev => ({ ...prev, error, loading: false }));
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    ...state,
    refetch: fetchAnalytics,
  };
}


// Export alias for compatibility
export { useKnowledgeEngine as useKnowledge };
