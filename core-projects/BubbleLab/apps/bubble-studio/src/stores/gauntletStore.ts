/**
 * Gauntlet Store
 * Manages gauntlet state and operations
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { Gauntlet, CreateGauntletRequest, UpdateGauntletRequest } from '../types/api';

// ============================================================================
// Gauntlet State Interface
// ============================================================================

interface GauntletState {
  // Data
  gauntlets: Record<string, Gauntlet>;
  gauntletIds: string[];
  selectedGauntletId: string | null;

  // Loading states
  isLoading: boolean;
  isCreating: boolean;
  isUpdating: boolean;
  isDeleting: boolean;

  // Error state
  error: string | null;

  // Actions
  setGauntlets: (gauntlets: Gauntlet[]) => void;
  setGauntlet: (gauntlet: Gauntlet) => void;
  removeGauntlet: (gauntletId: string) => void;
  setSelectedGauntletId: (gauntletId: string | null) => void;

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

export const useGauntletStore = create<GauntletState>()(
  subscribeWithSelector((set, get) => ({
    // Initial State
    gauntlets: {},
    gauntletIds: [],
    selectedGauntletId: null,
    isLoading: false,
    isCreating: false,
    isUpdating: false,
    isDeleting: false,
    error: null,

    // Set gauntlets (replace all)
    setGauntlets: (gauntlets) =>
      set(() => {
        const gauntletMap: Record<string, Gauntlet> = {};
        const ids: string[] = [];

        for (const gauntlet of gauntlets) {
          gauntletMap[gauntlet.id] = gauntlet;
          ids.push(gauntlet.id);
        }

        return {
          gauntlets: gauntletMap,
          gauntletIds: ids,
        };
      }),

    // Set single gauntlet (upsert)
    setGauntlet: (gauntlet) =>
      set((state) => {
        const gauntlets = { ...state.gauntlets };
        gauntlets[gauntlet.id] = gauntlet;

        const gauntletIds = state.gauntletIds.includes(gauntlet.id)
          ? state.gauntletIds
          : [...state.gauntletIds, gauntlet.id];

        return {
          gauntlets,
          gauntletIds,
        };
      }),

    // Remove gauntlet
    removeGauntlet: (gauntletId) =>
      set((state) => {
        const gauntlets = { ...state.gauntlets };
        delete gauntlets[gauntletId];

        return {
          gauntlets,
          gauntletIds: state.gauntletIds.filter((id) => id !== gauntletId),
          selectedGauntletId:
            state.selectedGauntletId === gauntletId
              ? null
              : state.selectedGauntletId,
        };
      }),

    // Set selected gauntlet
    setSelectedGauntletId: (gauntletId) =>
      set({ selectedGauntletId: gauntletId }),

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
        gauntlets: {},
        gauntletIds: [],
        selectedGauntletId: null,
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
 * Get gauntlet by ID
 */
export const getGauntletById = (id: string) => {
  return useGauntletStore.getState().gauntlets[id];
};

/**
 * Get all gauntlets
 */
export const getAllGauntlets = () => {
  const state = useGauntletStore.getState();
  return state.gauntletIds.map((id) => state.gauntlets[id]);
};

/**
 * Get selected gauntlet
 */
export const getSelectedGauntlet = () => {
  const state = useGauntletStore.getState();
  if (!state.selectedGauntletId) return null;
  return state.gauntlets[state.selectedGauntletId];
};

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook to get all gauntlets
 */
export const useGauntlets = () => {
  return useGauntletStore((state) => {
    return {
      gauntlets: state.gauntletIds.map((id) => state.gauntlets[id]),
      isLoading: state.isLoading,
      error: state.error,
    };
  });
};

/**
 * Hook to get a specific gauntlet
 */
export const useGauntlet = (gauntletId: string) => {
  return useGauntletStore((state) => {
    return {
      gauntlet: state.gauntlets[gauntletId] || null,
      isLoading: state.isLoading,
      error: state.error,
    };
  });
};

/**
 * Hook to get selected gauntlet
 */
export const useSelectedGauntlet = () => {
  const selectedGauntletId = useGauntletStore((state) => state.selectedGauntletId);
  const gauntlet = useGauntletStore((state) =>
    state.selectedGauntletId ? state.gauntlets[state.selectedGauntletId] : null
  );

  return {
    gauntlet,
    selectedGauntletId,
    setSelectedGauntletId: useGauntletStore((state) => state.setSelectedGauntletId),
  };
};
