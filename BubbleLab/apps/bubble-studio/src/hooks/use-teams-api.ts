/**
 * Team API Hooks
 * React hooks for team operations
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../lib/api-client';
import { useTeamStore } from '../stores/teamStore';
import { Team, CreateTeamRequest, UpdateTeamRequest, ListQueryParams } from '../types/api';

// ============================================================================
// Query Keys
// ============================================================================

export const teamKeys = {
  all: ['teams'] as const,
  lists: () => [...teamKeys.all, 'list'] as const,
  list: (params: ListQueryParams) => [...teamKeys.lists(), params] as const,
  details: () => [...teamKeys.all, 'detail'] as const,
  detail: (id: string) => [...teamKeys.details(), id] as const,
};

// ============================================================================
// Fetch Teams
// ============================================================================

export function useTeams(params?: ListQueryParams) {
  const setTeams = useTeamStore((state) => state.setTeams);
  const setIsLoading = useTeamStore((state) => state.setIsLoading);
  const setError = useTeamStore((state) => state.setError);

  return useQuery({
    queryKey: teamKeys.list(params || {}),
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const teams = await apiClient.getTeams(params);
        setTeams(teams);
        return teams;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch teams';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
  });
}

export function useTeam(teamId: string) {
  const setTeam = useTeamStore((state) => state.setTeam);
  const setIsLoading = useTeamStore((state) => state.setIsLoading);
  const setError = useTeamStore((state) => state.setError);

  return useQuery({
    queryKey: teamKeys.detail(teamId),
    queryFn: async () => {
      setIsLoading(true);
      setError(null);
      try {
        const team = await apiClient.getTeam(teamId);
        setTeam(team);
        return team;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch team';
        setError(message);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    enabled: !!teamId,
  });
}

// ============================================================================
// Create Team
// ============================================================================

export function useCreateTeam() {
  const queryClient = useQueryClient();
  const setIsCreating = useTeamStore((state) => state.setIsCreating);
  const setError = useTeamStore((state) => state.setError);
  const setTeam = useTeamStore((state) => state.setTeam);

  return useMutation({
    mutationFn: async (data: CreateTeamRequest) => {
      setIsCreating(true);
      setError(null);
      try {
        const team = await apiClient.createTeam(data);
        setTeam(team);
        return team;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to create team';
        setError(message);
        throw error;
      } finally {
        setIsCreating(false);
      }
    },
    onSuccess: () => {
      // Invalidate teams list query
      queryClient.invalidateQueries({ queryKey: teamKeys.lists() });
    },
  });
}

// ============================================================================
// Update Team
// ============================================================================

export function useUpdateTeam() {
  const queryClient = useQueryClient();
  const setIsUpdating = useTeamStore((state) => state.setIsUpdating);
  const setError = useTeamStore((state) => state.setError);
  const setTeam = useTeamStore((state) => state.setTeam);

  return useMutation({
    mutationFn: async ({ teamId, data }: { teamId: string; data: UpdateTeamRequest }) => {
      setIsUpdating(true);
      setError(null);
      try {
        const team = await apiClient.updateTeam(teamId, data);
        setTeam(team);
        return team;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to update team';
        setError(message);
        throw error;
      } finally {
        setIsUpdating(false);
      }
    },
    onSuccess: (_, variables) => {
      // Invalidate specific team query
      queryClient.invalidateQueries({
        queryKey: teamKeys.detail(variables.teamId),
      });

      // Invalidate teams list query
      queryClient.invalidateQueries({ queryKey: teamKeys.lists() });
    },
  });
}

// ============================================================================
// Delete Team
// ============================================================================

export function useDeleteTeam() {
  const queryClient = useQueryClient();
  const setIsDeleting = useTeamStore((state) => state.setIsDeleting);
  const setError = useTeamStore((state) => state.setError);
  const removeTeam = useTeamStore((state) => state.removeTeam);

  return useMutation({
    mutationFn: async (teamId: string) => {
      setIsDeleting(true);
      setError(null);
      try {
        await apiClient.deleteTeam(teamId);
        removeTeam(teamId);
        return teamId;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to delete team';
        setError(message);
        throw error;
      } finally {
        setIsDeleting(false);
      }
    },
    onSuccess: () => {
      // Invalidate teams list query
      queryClient.invalidateQueries({ queryKey: teamKeys.lists() });
    },
  });
}
