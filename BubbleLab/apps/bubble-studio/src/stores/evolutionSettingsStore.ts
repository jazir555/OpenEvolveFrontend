import { create } from 'zustand';
import {
  evolutionParameters,
  evolutionConstraintParameters,
  evolutionPreviewParameters,
  adversarialParameters,
  decompositionParameters,
  buildDefaultValues,
} from '@/lib/evolution/schemas';
import type { ParameterValue } from '@/types/evolution';

export interface EvolutionSnapshot {
  id: string;
  createdAt: string;
  evolutionInputs: Record<string, ParameterValue>;
  adversarialInputs: Record<string, ParameterValue>;
  decompositionInputs: Record<string, ParameterValue>;
}

interface EvolutionSettingsState {
  evolutionInputs: Record<string, ParameterValue>;
  adversarialInputs: Record<string, ParameterValue>;
  decompositionInputs: Record<string, ParameterValue>;
  snapshots: EvolutionSnapshot[];
  setEvolutionInput: (name: string, value: ParameterValue) => void;
  setAdversarialInput: (name: string, value: ParameterValue) => void;
  setDecompositionInput: (name: string, value: ParameterValue) => void;
  resetEvolution: () => void;
  resetAdversarial: () => void;
  resetDecomposition: () => void;
  resetAll: () => void;
  addSnapshot: () => void;
}

const evolutionDefaults = buildDefaultValues([
  ...evolutionParameters,
  ...evolutionConstraintParameters,
  ...evolutionPreviewParameters,
]);
const adversarialDefaults = buildDefaultValues(adversarialParameters);
const decompositionDefaults = buildDefaultValues(decompositionParameters);

const cloneDefaults = (defaults: Record<string, ParameterValue>) => ({
  ...defaults,
});

export const useEvolutionSettingsStore = create<EvolutionSettingsState>(
  (set, get) => ({
    evolutionInputs: cloneDefaults(evolutionDefaults),
    adversarialInputs: cloneDefaults(adversarialDefaults),
    decompositionInputs: cloneDefaults(decompositionDefaults),
    snapshots: [],
    setEvolutionInput: (name, value) =>
      set((state) => ({
        evolutionInputs: { ...state.evolutionInputs, [name]: value },
      })),
    setAdversarialInput: (name, value) =>
      set((state) => ({
        adversarialInputs: { ...state.adversarialInputs, [name]: value },
      })),
    setDecompositionInput: (name, value) =>
      set((state) => ({
        decompositionInputs: { ...state.decompositionInputs, [name]: value },
      })),
    resetEvolution: () =>
      set({ evolutionInputs: cloneDefaults(evolutionDefaults) }),
    resetAdversarial: () =>
      set({ adversarialInputs: cloneDefaults(adversarialDefaults) }),
    resetDecomposition: () =>
      set({ decompositionInputs: cloneDefaults(decompositionDefaults) }),
    resetAll: () =>
      set({
        evolutionInputs: cloneDefaults(evolutionDefaults),
        adversarialInputs: cloneDefaults(adversarialDefaults),
        decompositionInputs: cloneDefaults(decompositionDefaults),
      }),
    addSnapshot: () => {
      const now = new Date();
      const snapshot: EvolutionSnapshot = {
        id: `snapshot-${now.getTime()}`,
        createdAt: now.toISOString(),
        evolutionInputs: { ...get().evolutionInputs },
        adversarialInputs: { ...get().adversarialInputs },
        decompositionInputs: { ...get().decompositionInputs },
      };
      set((state) => ({
        snapshots: [snapshot, ...state.snapshots].slice(0, 20),
      }));
    },
  })
);
