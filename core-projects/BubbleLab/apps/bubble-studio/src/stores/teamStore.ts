/**
 * Team Store
 * Manages team state and operations
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { Team, CreateTeamRequest, UpdateTeamRequest } from '../types/api';

// ============================================================================
// Team State Interface
// ============================================================================

interface TeamState {
  // Data
  teams: Record<string, Team>;
  teamIds: string[];
  selectedTeamId: string | null;

  // Loading states
  isLoading: boolean;
  isCreating: boolean;
  isUpdating: boolean;
  isDeleting: boolean;

  // Error state
  error: string | null;

  // Actions
  setTeams: (teams: Team[]) => void;
  setTeam: (team: Team) => void;
  removeTeam: (teamId: string) => void;
  setSelectedTeamId: (teamId: string | null) => void;

  // Loading actions
  setIsLoading: (loading: boolean) => void;
  setIsCreating: (creating: boolean) => void;
  setIsUpdating: (updating: boolean) => void;
  setIsDeleting: (deleting: boolean) => void;

  // Error actions
  setError: (error: string | null) => void;
  clearError: () => void;

  // Reset
  reset: () => void;
}

// ============================================================================
// Create Store
// ============================================================================

export const useTeamStore = create<TeamState>()(
  subscribeWithSelector((set, get) => ({
    // Initial State
    teams: {},
    teamIds: [],
    selectedTeamId: null,
    isLoading: false,
    isCreating: false,
    isUpdating: false,
    isDeleting: false,
    error: null,

    // Set teams (replace all)
    setTeams: (teams) =>
      set(() => {
        const teamMap: Record<string, Team> = {};
        const ids: string[] = [];

        for (const team of teams) {
          teamMap[team.id] = team;
          ids.push(team.id);
        }

        return {
          teams: teamMap,
          teamIds: ids,
        };
      }),

    // Set single team (upsert)
    setTeam: (team) =>
      set((state) => {
        const teams = { ...state.teams };
        teams[team.id] = team;

        const teamIds = state.teamIds.includes(team.id)
          ? state.teamIds
          : [...state.teamIds, team.id];

        return {
          teams,
          teamIds,
        };
      }),

    // Remove team
    removeTeam: (teamId) =>
      set((state) => {
        const teams = { ...state.teams };
        delete teams[teamId];

        return {
          teams,
          teamIds: state.teamIds.filter((id) => id !== teamId),
          selectedTeamId:
            state.selectedTeamId === teamId ? null : state.selectedTeamId,
        };
      }),

    // Set selected team
    setSelectedTeamId: (teamId) =>
      set({ selectedTeamId: teamId }),

    // Loading actions
    setIsLoading: (isLoading) => set({ isLoading }),
    setIsCreating: (isCreating) => set({ isCreating }),
    setIsUpdating: (isUpdating) => set({ isUpdating }),
    setIsDeleting: (isDeleting) => set({ isDeleting }),

    // Error actions
    setError: (error) => set({ error }),
    clearError: () => set({ error: null }),

    // Reset
    reset: () =>
      set({
        teams: {},
        teamIds: [],
        selectedTeamId: null,
        isLoading: false,
        isCreating: false,
        isUpdating: false,
        isDeleting: false,
        error: null,
      }),
  }))
);

// ============================================================================
// Selectors
// ============================================================================

/**
 * Get team by ID
 */
export const getTeamById = (id: string) => {
  return useTeamStore.getState().teams[id];
};

/**
 * Get all teams
 */
export const getAllTeams = () => {
  const state = useTeamStore.getState();
  return state.teamIds.map((id) => state.teams[id]);
};

/**
 * Get selected team
 */
export const getSelectedTeam = () => {
  const state = useTeamStore.getState();
  if (!state.selectedTeamId) return null;
  return state.teams[state.selectedTeamId];
};

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook to get all teams
 */
export const useTeams = () => {
  return useTeamStore((state) => {
    return {
      teams: state.teamIds.map((id) => state.teams[id]),
      isLoading: state.isLoading,
      error: state.error,
    };
  });
};

/**
 * Hook to get a specific team
 */
export const useTeam = (teamId: string) => {
  return useTeamStore((state) => {
    return {
      team: state.teams[teamId] || null,
      isLoading: state.isLoading,
      error: state.error,
    };
  });
};

/**
 * Hook to get selected team
 */
export const useSelectedTeam = () => {
  const selectedTeamId = useTeamStore((state) => state.selectedTeamId);
  const team = useTeamStore((state) =>
    state.selectedTeamId ? state.teams[state.selectedTeamId] : null
  );

  return {
    team,
    selectedTeamId,
    setSelectedTeamId: useTeamStore((state) => state.setSelectedTeamId),
  };
};
