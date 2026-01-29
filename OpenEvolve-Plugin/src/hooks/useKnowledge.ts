/**
 * useKnowledge Hook
 *
 * Provides CRUD helpers and search over the knowledge content API.
 */

import { useCallback, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { contentApi } from '@/services/api';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import errorLogger from '@/utils/errorLogging';

export interface KnowledgeArtifact {
  artifact_id: string;
  title: string;
  content: string;
  language?: string;
  tags?: string[];
  created_at?: string;
  updated_at?: string;
}

const normalizeSearchQuery = (query: string) => query.trim().toLowerCase();

export function useKnowledge() {
  const queryClient = useQueryClient();

  const artifactsQuery = useQuery({
    queryKey: ['knowledge', 'artifacts'],
    queryFn: async () => {
      const response = await contentApi.list();
      return response.content as KnowledgeArtifact[];
    },
  });

  const createArtifactMutation = useMutation({
    mutationFn: async (data: Omit<KnowledgeArtifact, 'artifact_id'>) =>
      contentApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'artifacts'] });
    },
  });

  const updateArtifactMutation = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: Partial<KnowledgeArtifact> }) =>
      contentApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'artifacts'] });
    },
  });

  const deleteArtifactMutation = useMutation({
    mutationFn: async (id: string) => contentApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'artifacts'] });
    },
  });

  const searchArtifacts = useCallback(
    async (query: string) => {
      const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
        const normalized = normalizeSearchQuery(query);
        if (!normalized) {
          return artifactsQuery.data || [];
        }

        const fallbackResponse = await contentApi.list();
        const artifacts = artifactsQuery.data || fallbackResponse?.content || [];
        return artifacts.filter((artifact) => {
          const title = artifact.title?.toLowerCase() || '';
          const content = artifact.content?.toLowerCase() || '';
          return title.includes(normalized) || content.includes(normalized);
        });
      }, {
        strategy: 'retry',
        maxRetries: 2,
        retryDelay: 500,
        showUserNotification: false,
        logError: true,
        context: {
          component: 'useKnowledge',
          function: 'searchArtifacts',
          operation: 'SEARCH_KNOWLEDGE_ARTIFACTS',
          additionalData: { query }
        }
      });

      if (!result.success) {
        errorLogger.logError(result.error || 'Search artifacts failed', 'error', {
          component: 'useKnowledge',
          function: 'searchArtifacts',
          additionalData: { query, result }
        });
        return artifactsQuery.data || [];
      }

      return result.data!;
    },
    [artifactsQuery.data]
  );

  const artifacts = useMemo(() => artifactsQuery.data || [], [artifactsQuery.data]);

  return {
    artifacts,
    isLoading: artifactsQuery.isLoading,
    error: artifactsQuery.error,
    createArtifact: createArtifactMutation.mutateAsync,
    updateArtifact: updateArtifactMutation.mutateAsync,
    deleteArtifact: deleteArtifactMutation.mutateAsync,
    searchArtifacts,
    isCreating: createArtifactMutation.isPending,
    isUpdating: updateArtifactMutation.isPending,
    isDeleting: deleteArtifactMutation.isPending,
  };
}
