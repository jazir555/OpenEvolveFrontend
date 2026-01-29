import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { errorLogger } from '@/utils';

export type SettingsScope = 'Global' | 'Provider' | 'Model';

export interface GenerationSettings {
  temperature: number;
  top_p: number;
  max_tokens: number;
  frequency_penalty: number;
  presence_penalty: number;
  seed?: number | null;
  reasoning_effort?: 'low' | 'medium' | 'high';
  api_timeout?: number;
  api_retries?: number;
}

export interface EvolutionSettings {
  max_iterations: number;
  population_size: number;
  num_islands?: number;
  migration_interval?: number;
  migration_rate?: number;
  archive_size?: number;
  elite_ratio?: number;
  exploration_ratio?: number;
  exploitation_ratio?: number;
  checkpoint_interval?: number;
  feature_dimensions?: string[];
  feature_bins?: number;
  diversity_metric?: string;
}

export interface ScopedSettings {
  generation: GenerationSettings;
  evolution: EvolutionSettings;
}

export interface ProviderSettings {
  settings: ScopedSettings;
  models: Record<string, ScopedSettings>;
  apiKeyLastFour?: string;
}

export interface ParameterSettings {
  global: ScopedSettings;
  providers: Record<string, ProviderSettings>;
}

interface SettingsState {
  scope: SettingsScope;
  provider?: string;
  model?: string;
  parameterSettings: ParameterSettings;

  setScope: (scope: SettingsScope) => void;
  setProvider: (provider?: string) => void;
  setModel: (model?: string) => void;

  updateGlobalSettings: (settings: Partial<ScopedSettings>) => void;
  updateProviderSettings: (provider: string, settings: Partial<ScopedSettings>) => void;
  updateModelSettings: (provider: string, model: string, settings: Partial<ScopedSettings>) => void;
  setProviderApiKeyLastFour: (provider: string, lastFour?: string) => void;
}

const defaultGeneration: GenerationSettings = {
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 2048,
  frequency_penalty: 0,
  presence_penalty: 0,
  seed: null,
  reasoning_effort: 'medium',
  api_timeout: 30,
  api_retries: 3,
};

const defaultEvolution: EvolutionSettings = {
  max_iterations: 100,
  population_size: 50,
  num_islands: 5,
  migration_interval: 50,
  migration_rate: 0.1,
  archive_size: 100,
  elite_ratio: 0.1,
  exploration_ratio: 0.2,
  exploitation_ratio: 0.7,
  checkpoint_interval: 100,
  feature_dimensions: [],
  feature_bins: 10,
  diversity_metric: 'edit_distance',
};

const defaultScopedSettings: ScopedSettings = {
  generation: defaultGeneration,
  evolution: defaultEvolution,
};

export const useSettingsStore = create<SettingsState>()(
  devtools(
    persist(
      (set, get) => ({
        scope: 'Global',
        provider: undefined,
        model: undefined,
        parameterSettings: {
          global: defaultScopedSettings,
          providers: {},
        },

        setScope: (scope) => {
          try {
            set({ scope });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'setScope', additionalData: { scope } }
            );
          }
        },

        setProvider: (provider) => {
          try {
            set({ provider, model: undefined });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'setProvider', additionalData: { provider } }
            );
          }
        },

        setModel: (model) => {
          try {
            set({ model });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'setModel', additionalData: { model } }
            );
          }
        },

        updateGlobalSettings: (settings) => {
          try {
            set((state) => ({
              parameterSettings: {
                ...state.parameterSettings,
                global: {
                  generation: {
                    ...state.parameterSettings.global.generation,
                    ...settings.generation,
                  },
                  evolution: {
                    ...state.parameterSettings.global.evolution,
                    ...settings.evolution,
                  },
                },
              },
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'updateGlobalSettings', additionalData: { settings } }
            );
          }
        },

        updateProviderSettings: (provider, settings) => {
          try {
            set((state) => {
              const currentProvider = state.parameterSettings.providers[provider] || {
                settings: defaultScopedSettings,
                models: {},
              };
              return {
                parameterSettings: {
                  ...state.parameterSettings,
                  providers: {
                    ...state.parameterSettings.providers,
                    [provider]: {
                      ...currentProvider,
                      settings: {
                        generation: {
                          ...currentProvider.settings.generation,
                          ...settings.generation,
                        },
                        evolution: {
                          ...currentProvider.settings.evolution,
                          ...settings.evolution,
                        },
                      },
                    },
                  },
                },
              };
            });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'updateProviderSettings', additionalData: { provider, settings } }
            );
          }
        },

        updateModelSettings: (provider, model, settings) => {
          try {
            set((state) => {
              const currentProvider = state.parameterSettings.providers[provider] || {
                settings: defaultScopedSettings,
                models: {},
              };
              const currentModelSettings = currentProvider.models[model] || defaultScopedSettings;
              return {
                parameterSettings: {
                  ...state.parameterSettings,
                  providers: {
                    ...state.parameterSettings.providers,
                    [provider]: {
                      ...currentProvider,
                      models: {
                        ...currentProvider.models,
                        [model]: {
                          generation: {
                            ...currentModelSettings.generation,
                            ...settings.generation,
                          },
                          evolution: {
                            ...currentModelSettings.evolution,
                            ...settings.evolution,
                          },
                        },
                      },
                    },
                  },
                },
              };
            });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'updateModelSettings', additionalData: { provider, model, settings } }
            );
          }
        },

        setProviderApiKeyLastFour: (provider, lastFour) => {
          try {
            set((state) => {
              const currentProvider = state.parameterSettings.providers[provider] || {
                settings: defaultScopedSettings,
                models: {},
              };
              return {
                parameterSettings: {
                  ...state.parameterSettings,
                  providers: {
                    ...state.parameterSettings.providers,
                    [provider]: {
                      ...currentProvider,
                      apiKeyLastFour: lastFour,
                    },
                  },
                },
              };
            });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'SettingsStore', function: 'setProviderApiKeyLastFour', additionalData: { provider, lastFour } }
            );
          }
        },
      }),
      {
        name: 'openevolve-settings',
        partialize: (state) => ({
          scope: state.scope,
          provider: state.provider,
          model: state.model,
          parameterSettings: state.parameterSettings,
        }),
      }
    ),
    { name: 'SettingsStore' }
  )
);
