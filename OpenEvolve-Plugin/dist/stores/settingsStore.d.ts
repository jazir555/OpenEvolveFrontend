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
export declare const useSettingsStore: import('zustand').UseBoundStore<Omit<Omit<import('zustand').StoreApi<SettingsState>, "setState"> & {
    setState<A extends string | {
        type: string;
    }>(partial: SettingsState | Partial<SettingsState> | ((state: SettingsState) => SettingsState | Partial<SettingsState>), replace?: boolean, action?: A): void;
}, "persist"> & {
    persist: {
        setOptions: (options: Partial<import('zustand/middleware').PersistOptions<SettingsState, {
            scope: SettingsScope;
            provider: string;
            model: string;
            parameterSettings: ParameterSettings;
        }>>) => void;
        clearStorage: () => void;
        rehydrate: () => Promise<void> | void;
        hasHydrated: () => boolean;
        onHydrate: (fn: (state: SettingsState) => void) => () => void;
        onFinishHydration: (fn: (state: SettingsState) => void) => () => void;
        getOptions: () => Partial<import('zustand/middleware').PersistOptions<SettingsState, {
            scope: SettingsScope;
            provider: string;
            model: string;
            parameterSettings: ParameterSettings;
        }>>;
    };
}>;
export {};
