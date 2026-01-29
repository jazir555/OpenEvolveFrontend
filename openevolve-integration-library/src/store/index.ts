import { create } from 'zustand';
import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';
import { IntegrationError, createIntegrationError } from '../api/errors';

export interface OpenEvolveState {
  client: OpenEvolveClient | null;
  results: Record<string, any>;
  loading: Record<string, boolean>;
  errors: Record<string, IntegrationError | null>;
  versions: Record<string, number>;
  
  initialize: (client: OpenEvolveClient) => void;
  execute: (integration: IntegrationName | string, inputs: any) => Promise<any>;
  clearResult: (integration: string) => void;
  reset: () => void;
}

export const createOpenEvolveStore = () => create<OpenEvolveState>((set, get) => ({
  client: null,
  results: {},
  loading: {},
  errors: {},
  versions: {},

  initialize: (client: OpenEvolveClient) => set({ client }),

  execute: async (integration: IntegrationName | string, inputs: any) => {
    const { client, versions } = get();
    const key = integration as string;
    
    if (!client) {
      const initError = createIntegrationError(key, new Error('Client not initialized'));
      set((state) => ({
        errors: { ...state.errors, [key]: initError }
      }));
      throw initError;
    }

    const version = (versions[key] || 0) + 1;

    set((state) => ({
      loading: { ...state.loading, [key]: true },
      errors: { ...state.errors, [key]: null },
      versions: { ...state.versions, [key]: version }
    }));

    try {
      const result = await client.execute(key, inputs);
      
      // Update only if this is still the current version
      if (get().versions[key] === version) {
        set((state) => ({
          results: { ...state.results, [key]: result },
          loading: { ...state.loading, [key]: false }
        }));
      }
      return result;
    } catch (error) {
      const integrationError = createIntegrationError(key, error);
      if (get().versions[key] === version) {
        set((state) => ({
          errors: { ...state.errors, [key]: integrationError },
          loading: { ...state.loading, [key]: false }
        }));
      }
      throw integrationError;
    }
  },


  clearResult: (integration: string) => set((state) => {
    const results = { ...state.results };
    const loading = { ...state.loading };
    const errors = { ...state.errors };
    const versions = { ...state.versions };
    
    delete results[integration];
    delete loading[integration];
    delete errors[integration];
    delete versions[integration];
    
    return { results, loading, errors, versions };
  }),

  reset: () => set({
    results: {},
    loading: {},
    errors: {},
    versions: {}
  })
}));
