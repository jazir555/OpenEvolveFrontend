// @ts-nocheck
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as api from '@/services/api/endpoints';
import { useKnowledgeStore } from '@/stores/knowledgeStore';

/**
 * Knowledge base operations hook
 */
export function useKnowledge() {
  const queryClient = useQueryClient();
  const { setArtifacts, addArtifact, updateArtifact, removeArtifact } = useKnowledgeStore();

  // Fetch all artifacts
  const { data, isLoading, error } = useQuery({
    queryKey: ['knowledge'],
    queryFn: async () => {
      const response = await api.content.list();
      const content = response?.content || [];
      setArtifacts(content);
      return content;
    },
    staleTime: 60000,
  });

  // Create artifact mutation
  const createArtifact = useMutation({
    mutationFn: async (data: {
      title: string;
      content: string;
      language?: string;
      tags?: string[];
    }) => {
      const response = await api.content.create(data);
      addArtifact(response);
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge'] });
    },
  });

  // Update artifact mutation
  const updateArtifactMutation = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: any }) => {
      const response = await api.content.update(id, data);
      updateArtifact(id, response);
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge'] });
    },
  });

  // Delete artifact mutation
  const deleteArtifactMutation = useMutation({
    mutationFn: async (id: string) => {
      await api.content.delete(id);
      removeArtifact(id);
      return id;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge'] });
    },
  });

  return {
    artifacts: data || [],
    isLoading,
    error,
    createArtifact: createArtifact.mutateAsync,
    updateArtifact: updateArtifactMutation.mutateAsync,
    deleteArtifact: deleteArtifactMutation.mutateAsync,
    isCreating: createArtifact.isPending,
    isUpdating: updateArtifactMutation.isPending,
    isDeleting: deleteArtifactMutation.isPending,
  };
}

/**
 * Single artifact hook
 */
export function useArtifact(artifactId?: string) {
  const queryClient = useQueryClient();
  const { selectedArtifact, setSelectedArtifact } = useKnowledgeStore();

  const { data, isLoading, error } = useQuery({
    queryKey: ['artifact', artifactId],
    queryFn: async () => {
      if (!artifactId) throw new Error('Artifact ID is required');
      const response = await api.content.getById(artifactId);
      setSelectedArtifact(response);
      return response;
    },
    enabled: !!artifactId,
  });

  return {
    artifact: data || selectedArtifact,
    isLoading,
    error,
    refetch: () => queryClient.invalidateQueries({ queryKey: ['artifact', artifactId] }),
  };
}

/**
 * Artifact versions hook
 */
export function useArtifactVersions(artifactId?: string) {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ['artifact', artifactId, 'versions'],
    queryFn: async () => {
      if (!artifactId) throw new Error('Artifact ID is required');
      return await api.version.getHistory(artifactId);
    },
    enabled: !!artifactId,
  });
}

/**
 * Artifact diff hook
 */
export function useArtifactDiff(artifactId?: string, version1?: number, version2?: number) {
  return useQuery({
    queryKey: ['artifact', artifactId, 'diff', version1, version2],
    queryFn: async () => {
      if (!artifactId || !version1 || !version2) {
        throw new Error('Artifact ID and both versions are required');
      }
      return await api.version.getDiff(artifactId, version1, version2);
    },
    enabled: !!artifactId && !!version1 && !!version2,
  });
}

/**
 * Knowledge search hook
 */
export function useKnowledgeSearch(query: string, filters?: {
  tags?: string[];
  language?: string;
}) {
  const { searchResults } = useKnowledgeStore();

  // Fetch all artifacts and filter client-side
  // (In production, this would be a server-side search endpoint)
  const { data, isLoading } = useQuery({
    queryKey: ['knowledge', 'search', query, filters],
    queryFn: async () => {
      const response = await api.content.list();

      let filtered = response?.content || [];

      // Filter by query
      if (query) {
        const lowerQuery = query.toLowerCase();
        filtered = filtered.filter((artifact) => {
          const title = artifact.title?.toLowerCase() || '';
          const content = artifact.content?.toLowerCase() || '';
          return title.includes(lowerQuery) || content.includes(lowerQuery);
        });
      }

      // Filter by tags
      if (filters?.tags && filters.tags.length > 0) {
        filtered = filtered.filter((artifact) =>
          filters.tags!.some((tag) => (artifact.tags || []).includes(tag))
        );
      }

      // Filter by language
      if (filters?.language) {
        filtered = filtered.filter(artifact =>
          artifact.language === filters.language
        );
      }

      return filtered;
    },
    staleTime: 30000,
  });

  return {
    results: data || [],
    isLoading,
    total: data?.length || 0,
  };
}

/**
 * Knowledge graph hook
 */
export function useKnowledgeGraph() {
  const { graphData, setGraphData, isLoading, setLoading } = useKnowledgeStore();

  const { data: artifacts } = useQuery({
    queryKey: ['knowledge'],
    queryFn: async () => {
      const response = await api.content.list();
      return response?.content || [];
    },
  });

  // Build graph from artifacts
  const buildGraph = () => {
    if (!Array.isArray(artifacts)) return;

    const nodes = artifacts.map(artifact => ({
      id: artifact.artifact_id,
      label: artifact.title,
      type: artifact.language || 'unknown',
      data: artifact,
    }));

    const edges: Array<{
      id: string;
      source: string;
      target: string;
      type: string;
    }> = [];

    // Create edges based on shared tags
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const node1 = nodes[i];
        const node2 = nodes[j];
        const sharedTags = (node1.data.tags || []).filter((tag: string) =>
          (node2.data.tags || []).includes(tag)
        );

        if (sharedTags.length > 0) {
          edges.push({
            id: `${node1.id}-${node2.id}`,
            source: node1.id,
            target: node2.id,
            type: 'shared_tag',
          });
        }
      }
    }

    setGraphData({ nodes, edges });
  };

  return {
    graphData,
    isLoading,
    buildGraph,
  };
}

/**
 * Artifact comments hook
 */
export function useArtifactComments(artifactId?: string) {
  const queryClient = useQueryClient();

  // Get comments
  const { data, isLoading, error } = useQuery({
    queryKey: ['artifact', artifactId, 'comments'],
    queryFn: async () => {
      if (!artifactId) throw new Error('Artifact ID is required');
      return await api.comments.get(artifactId);
    },
    enabled: !!artifactId,
  });

  // Add comment mutation
  const addComment = useMutation({
    mutationFn: async (data: {
      comment: string;
      line_start?: number;
      line_end?: number;
      parent_comment_id?: string;
    }) => {
      if (!artifactId) throw new Error('Artifact ID is required');
      return await api.comments.add(artifactId, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['artifact', artifactId, 'comments'] });
    },
  });

  return {
    comments: data || [],
    isLoading,
    error,
    addComment: addComment.mutateAsync,
    isAdding: addComment.isPending,
  };
}

/**
 * Collaboration hook
 */
export function useCollaboration(contentId?: string) {
  const queryClient = useQueryClient();

  // Create room mutation
  const createRoom = useMutation({
    mutationFn: async (data?: { room_name?: string }) => {
      if (!contentId) throw new Error('Content ID is required');
      return await api.collaboration.createRoom({
        content_id: contentId,
        ...data,
      });
    },
  });

  // Get room users
  const getRoomUsers = useQuery({
    queryKey: ['collaboration', 'room', 'users'],
    queryFn: async () => {
      // This would be called after creating a room
      return [];
    },
    enabled: false,
  });

  return {
    createRoom: createRoom.mutateAsync,
    users: getRoomUsers.data || [],
    isCreatingRoom: createRoom.isPending,
    isLoadingUsers: getRoomUsers.isLoading,
  };
}
