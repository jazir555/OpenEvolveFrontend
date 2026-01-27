/**
 * Gauntlet API Hooks
 * React hooks for gauntlet operations
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../lib/api-client';
import { useGauntletStore } from '../stores/gauntletStore';
import { Gauntlet, CreateGauntletRequest, UpdateGauntletRequest, ListQueryParams } from '../types/api';

// ============================================================================
// Query Keys
// ============================================================================

export const gauntletKeys = {
  all: ['gauntlets'] as const,
  lists: () => [...gauntletKeys.all, 'list'] as const,
  list: (params: ListQueryParams) => [...gauntletKeys.lists(), params] as const,
  details: () => [...gauntletKeys.all, 'detail'] as const,
  detail: (id: string) => [...gauntletKeys.details(), id] as const,
};

// ============================================================================
// Fetch Gauntlets
// ============================================================================

export function useGauntlets(params?: ListQueryParams) {
  const setGauntlets = useGauntletStore((state) => state.setGauntlets);
  const setIsLoading = useGauntletStore((state) => state.setIsLoading);
  const setError = useGauntletStore((state) => state.setError);

  return useQuery({
    queryKey: gauntletKeys.list(params || {}),
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const gauntlets = await apiClient.getGauntlets(params);
        setGauntlets(gauntlets);
        return gauntlets;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch gauntlets';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
  });
}

export function useGauntlet(gauntletId: string) {
  const setGauntlet = useGauntletStore((state) => state.setGauntlet);
  const setIsLoading = useGauntletStore((state) => state.setIsLoading);
  const setError = useGauntletStore((state) => state.setError);

  return useQuery({
    queryKey: gauntletKeys.detail(gauntletId),
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const gauntlet = await apiClient.getGauntlet(gauntletId);
        setGauntlet(gauntlet);
        return gauntlet;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch gauntlet';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    enabled: !!gauntletId,
  });
}

// ============================================================================
// Create Gauntlet
// ============================================================================

export function useCreateGauntlet() {
  const queryClient = useQueryClient();
  const setIsCreating = useGauntletStore((state) => state.setIsCreating);
  const setError = useGauntletStore((state) => state.setError);
  const setGauntlet = useGauntletStore((state) => state.setGauntlet);

  return useMutation({
    mutationFn: async (data: CreateGauntletRequest) => {
      setIsCreating(true);
      setError(null);
      try {
        const gauntlet = await apiClient.createGauntlet(data);
        setGauntlet(gauntlet);
        return gauntlet;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to create gauntlet';
        setError(message);
        throw error;
      } finally {
        setIsCreating(false);
      }
    },
    onSuccess: () => {
      // Invalidate gauntlets list query
      queryClient.invalidateQueries({ queryKey: gauntletKeys.lists() });
    },
  });
}

// ============================================================================
// Update Gauntlet
// ============================================================================

export function useUpdateGauntlet() {
  const queryClient = useQueryClient();
  const setIsUpdating = useGauntletStore((state) => state.setIsUpdating);
  const setError = useGauntletStore((state) => state.setError);
  const setGauntlet = useGauntletStore((state) => state.setGauntlet);

  return useMutation({
    mutationFn: async ({ gauntletId, data }: { gauntletId: string; data: UpdateGauntletRequest }) => {
      setIsUpdating(true);
      setError(null);
      try {
        const gauntlet = await apiClient.updateGauntlet(gauntletId, data);
        setGauntlet(gauntlet);
        return gauntlet;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to update gauntlet';
        setError(message);
        throw error;
      } finally {
        setIsUpdating(false);
      }
    },
    onSuccess: (_: any, variables: any) => {
      // Invalidate specific gauntlet query
      queryClient.invalidateQueries({
        queryKey: gauntletKeys.detail(variables.gauntletId),
      });

      // Invalidate gauntlets list query
      queryClient.invalidateQueries({ queryKey: gauntletKeys.lists() });
    },
  });
}

// ============================================================================
// Delete Gauntlet
// ============================================================================

export function useDeleteGauntlet() {
  const queryClient = useQueryClient();
  const setIsDeleting = useGauntletStore((state) => state.setIsDeleting);
  const setError = useGauntletStore((state) => state.setError);
  const removeGauntlet = useGauntletStore((state) => state.removeGauntlet);

  return useMutation({
    mutationFn: async (gauntletId: string) => {
      setIsDeleting(true);
      setError(null);
      try {
        await apiClient.deleteGauntlet(gauntletId);
        removeGauntlet(gauntletId);
        return gauntletId;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to delete gauntlet';
        setError(message);
        throw error;
      } finally {
        setIsDeleting(false);
      }
    },
    onSuccess: () => {
      // Invalidate gauntlets list query
      queryClient.invalidateQueries({ queryKey: gauntletKeys.lists() });
    },
  });
}
