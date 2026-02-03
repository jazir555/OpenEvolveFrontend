import { create } from 'zustand';
import type { EvolutionWebSocketMessage } from '@/types/evolution';

export type EvolutionRunStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'paused'
  | 'completed'
  | 'stopped'
  | 'error';

export type EvolutionSocketStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error';

interface EvolutionRuntimeState {
  evolutionId: string | null;
  websocketUrl: string | null;
  status: EvolutionRunStatus;
  socketStatus: EvolutionSocketStatus;
  reconnectToken: number;
  progress: number;
  statusMessage: string;
  latestGeneration: number | null;
  latestFitness: number | null;
  error: string | null;
  lastEvent: EvolutionWebSocketMessage | null;
  setStart: (evolutionId: string, websocketUrl: string) => void;
  setSocketStatus: (status: EvolutionSocketStatus) => void;
  triggerReconnect: () => void;
  setProgress: (progress: number, message: string) => void;
  setGeneration: (generation: number, fitness: number) => void;
  setStatus: (status: EvolutionRunStatus) => void;
  setError: (message: string) => void;
  clearError: () => void;
  setLastEvent: (event: EvolutionWebSocketMessage) => void;
  reset: () => void;
}

const initialState: Omit<
  EvolutionRuntimeState,
  | 'setStart'
  | 'setSocketStatus'
  | 'setProgress'
  | 'setGeneration'
  | 'setStatus'
  | 'setError'
  | 'setLastEvent'
  | 'clearError'
  | 'reset'
  | 'triggerReconnect'
> = {
  evolutionId: null,
  websocketUrl: null,
  status: 'idle',
  socketStatus: 'disconnected',
  reconnectToken: 0,
  progress: 0,
  statusMessage: 'Idle',
  latestGeneration: null,
  latestFitness: null,
  error: null,
  lastEvent: null,
};

export const useEvolutionRuntimeStore = create<EvolutionRuntimeState>((set) => ({
  ...initialState,
  setStart: (evolutionId, websocketUrl) =>
    set({
      evolutionId,
      websocketUrl,
      status: 'running',
      socketStatus: 'connecting',
      progress: 0,
      statusMessage: 'Starting evolution...',
      latestGeneration: null,
      latestFitness: null,
      error: null,
    }),
  setSocketStatus: (socketStatus) => set({ socketStatus }),
  triggerReconnect: () =>
    set((state) => ({ reconnectToken: state.reconnectToken + 1 })),
  setProgress: (progress, message) =>
    set({
      progress,
      statusMessage: message,
      status: progress >= 100 ? 'completed' : 'running',
    }),
  setGeneration: (generation, fitness) =>
    set({
      latestGeneration: generation,
      latestFitness: fitness,
    }),
  setStatus: (status) => set({ status }),
  setError: (message) =>
    set({
      status: 'error',
      error: message,
      statusMessage: message,
      socketStatus: 'error',
    }),
  clearError: () =>
    set({
      error: null,
      statusMessage: 'Running',
    }),
  setLastEvent: (event) => set({ lastEvent: event }),
  reset: () => set({ ...initialState }),
}));
