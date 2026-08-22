/**
 * Configuration Store
 * Manages LLM configuration and application settings
 * Replaces legacy session state for configuration
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import {
  LLMProvider,
  LLMConfig,
  UIState,
  PanelMode,
  DeterminismDefaults,
  DecompositionDefaults,
} from '../types/api';

// ============================================================================
// Configuration State Interface
// ============================================================================

interface ConfigState extends LLMConfig {
  // ICR Configuration
  auto_refine_enabled: boolean;
  reward_calibration_enabled: boolean;
  reward_calibration_threshold: number;
  heatmap_analysis_enabled: boolean;
  heatmap_snapshot_interval: number;
  vlm_provider?: string;
  vlm_model?: string;

  // Integration Defaults
  determinism_defaults: DeterminismDefaults;
  decomposition_defaults: DecompositionDefaults;

  // UI State
  ui: UIState;

  // Actions
  setLLMProvider: (provider: LLMProvider) => void;
  setApiKey: (apiKey: string) => void;
  setBaseUrl: (baseUrl: string) => void;
  setModelLeanAide: (model: string) => void;
  setModelText: (model: string) => void;
  setModelImg: (model: string) => void;
  setTemperature: (temp: number) => void;
  setTopP: (topP: number) => void;
  setMaxTokens: (tokens: number) => void;
  setFrequencyPenalty: (penalty: number) => void;
  setPresencePenalty: (penalty: number) => void;

  // ICR Actions
  setAutoRefineEnabled: (enabled: boolean) => void;
  setRewardCalibrationEnabled: (enabled: boolean) => void;
  setRewardCalibrationThreshold: (threshold: number) => void;
  setHeatmapAnalysisEnabled: (enabled: boolean) => void;
  setHeatmapSnapshotInterval: (interval: number) => void;
  setVlmProvider: (provider: string) => void;
  setVlmModel: (model: string) => void;

  // Defaults Actions
  setDeterminismDefaults: (defaults: DeterminismDefaults) => void;
  setDecompositionDefaults: (defaults: DecompositionDefaults) => void;

  // UI Actions
  setSelectedFlowId: (flowId: string | null) => void;
  setPanelMode: (mode: UIState['panelMode']) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setDarkMode: (darkMode: boolean) => void;
  setFontSize: (size: UIState['fontSize']) => void;
  setAutoSave: (autoSave: boolean) => void;

  // Bulk Actions
  setLLMConfig: (config: Partial<LLMConfig>) => void;
  setICRConfig: (config: Partial<Pick<ConfigState,
    | 'auto_refine_enabled'
    | 'reward_calibration_enabled'
    | 'reward_calibration_threshold'
    | 'heatmap_analysis_enabled'
    | 'heatmap_snapshot_interval'
    | 'vlm_provider'
    | 'vlm_model'
  >>) => void;
  setDefaultsConfig: (config: Partial<Pick<ConfigState,
    | 'determinism_defaults'
    | 'decomposition_defaults'
  >>) => void;
  setUIState: (state: Partial<UIState>) => void;

  // Reset
  resetLLMConfig: () => void;
  resetICRConfig: () => void;
  resetDefaultsConfig: () => void;
  resetUIState: () => void;
}

// ============================================================================
// Default Configuration
// ============================================================================

const defaultLLMConfig: LLMConfig = {
  provider: LLMProvider.OPENAI,
  api_key: '',
  base_url: undefined,
  model_leanaide: 'gpt-4',
  model_text: 'gpt-3.5-turbo',
  model_img: 'gpt-4-vision-preview',
  temperature: 0.7,
  top_p: 1.0,
  max_tokens: 2000,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
};

const defaultUIState: UIState = {
  selectedFlowId: null,
  panelMode: PanelMode.BUBBLE_LIST,
  sidebarCollapsed: false,
  darkMode: false,
  fontSize: 'medium',
  autoSave: true,
};

const defaultICRConfig = {
  auto_refine_enabled: false,
  reward_calibration_enabled: true,
  reward_calibration_threshold: 0.6,
  heatmap_analysis_enabled: true,
  heatmap_snapshot_interval: 10,
  vlm_provider: '',
  vlm_model: '',
};

const defaultDeterminismDefaults: DeterminismDefaults = {
  mode: 'auto',
  cloud_provider: '',
  cloud_model: '',
  cloud_base_url: '',
  local_provider: 'hf',
  local_model: '',
  local_device: 'cpu',
  local_dtype: 'auto',
  config: {},
  detllm_backend: '',
  detllm_model: '',
  check_tier: 2,
  check_runs: 3,
  check_provider: '',
  check_model: '',
  check_base_url: '',
  check_device: 'cpu',
  check_dtype: 'auto',
};

const defaultDecompositionDefaults: DecompositionDefaults = {
  strategy: '',
  enable_adaptive_selection: true,
  maker_config: {},
  openevolve_client_config: {},
  mdap_enabled: false,
  mdap_config: {},
  maker_enabled: false,
  workflow_max_refinement_loops: 3,
};

// ============================================================================
// Create Store
// ============================================================================

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      // Initial State
      ...defaultLLMConfig,
      ...defaultICRConfig,
      determinism_defaults: defaultDeterminismDefaults,
      decomposition_defaults: defaultDecompositionDefaults,
      ui: defaultUIState,

      // LLM Configuration Actions
      setLLMProvider: (provider) =>
        set({ provider }),

      setApiKey: (api_key) =>
        set({ api_key }),

      setBaseUrl: (base_url) =>
        set({ base_url }),

      setModelLeanAide: (model_leanaide) =>
        set({ model_leanaide }),

      setModelText: (model_text) =>
        set({ model_text }),

      setModelImg: (model_img) =>
        set({ model_img }),

      setTemperature: (temperature) =>
        set({ temperature }),

      setTopP: (top_p) =>
        set({ top_p }),

      setMaxTokens: (max_tokens) =>
        set({ max_tokens }),

      setFrequencyPenalty: (frequency_penalty) =>
        set({ frequency_penalty }),

      setPresencePenalty: (presence_penalty) =>
        set({ presence_penalty }),

      // ICR Configuration Actions
      setAutoRefineEnabled: (auto_refine_enabled) =>
        set({ auto_refine_enabled }),

      setRewardCalibrationEnabled: (reward_calibration_enabled) =>
        set({ reward_calibration_enabled }),

      setRewardCalibrationThreshold: (reward_calibration_threshold) =>
        set({ reward_calibration_threshold }),

      setHeatmapAnalysisEnabled: (heatmap_analysis_enabled) =>
        set({ heatmap_analysis_enabled }),

      setHeatmapSnapshotInterval: (heatmap_snapshot_interval) =>
        set({ heatmap_snapshot_interval }),

      setVlmProvider: (vlm_provider) =>
        set({ vlm_provider }),

      setVlmModel: (vlm_model) =>
        set({ vlm_model }),

      // Defaults Actions
      setDeterminismDefaults: (determinism_defaults) =>
        set({ determinism_defaults }),

      setDecompositionDefaults: (decomposition_defaults) =>
        set({ decomposition_defaults }),

      // UI State Actions
      setSelectedFlowId: (selectedFlowId) =>
        set((state) => ({
          ui: { ...state.ui, selectedFlowId }
        })),

      setPanelMode: (panelMode) =>
        set((state) => ({
          ui: { ...state.ui, panelMode }
        })),

      setSidebarCollapsed: (sidebarCollapsed) =>
        set((state) => ({
          ui: { ...state.ui, sidebarCollapsed }
        })),

      setDarkMode: (darkMode) =>
        set((state) => ({
          ui: { ...state.ui, darkMode }
        })),

      setFontSize: (fontSize) =>
        set((state) => ({
          ui: { ...state.ui, fontSize }
        })),

      setAutoSave: (autoSave) =>
        set((state) => ({
          ui: { ...state.ui, autoSave }
        })),

      // Bulk Actions
      setLLMConfig: (config) =>
        set((state) => ({
          ...state,
          ...config
        })),

      setICRConfig: (config) =>
        set((state) => ({
          ...state,
          ...config
        })),

      setDefaultsConfig: (config) =>
        set((state) => ({
          ...state,
          ...config
        })),

      setUIState: (uiState) =>
        set((state) => ({
          ui: { ...state.ui, ...uiState }
        })),

      // Reset Actions
      resetLLMConfig: () =>
        set((state) => ({
          ...state,
          ...defaultLLMConfig
        })),

      resetICRConfig: () =>
        set((state) => ({
          ...state,
          ...defaultICRConfig
        })),

      resetDefaultsConfig: () =>
        set((state) => ({
          ...state,
          determinism_defaults: defaultDeterminismDefaults,
          decomposition_defaults: defaultDecompositionDefaults,
        })),

      resetUIState: () =>
        set((state) => ({
          ui: defaultUIState
        })),
    }),
    {
      name: 'openevolve-config-storage',
      storage: createJSONStorage(() => localStorage),
      // Only persist certain fields (exclude sensitive api_key from partial persistence)
      partialize: (state) => ({
        // LLM Config (exclude api_key for security)
        provider: state.provider,
        base_url: state.base_url,
        model_leanaide: state.model_leanaide,
        model_text: state.model_text,
        model_img: state.model_img,
        temperature: state.temperature,
        top_p: state.top_p,
        max_tokens: state.max_tokens,
        frequency_penalty: state.frequency_penalty,
        presence_penalty: state.presence_penalty,

        // ICR Config
        auto_refine_enabled: state.auto_refine_enabled,
        reward_calibration_enabled: state.reward_calibration_enabled,
        reward_calibration_threshold: state.reward_calibration_threshold,
        heatmap_analysis_enabled: state.heatmap_analysis_enabled,
        heatmap_snapshot_interval: state.heatmap_snapshot_interval,
        vlm_provider: state.vlm_provider,
        vlm_model: state.vlm_model,

        // Defaults
        determinism_defaults: state.determinism_defaults,
        decomposition_defaults: state.decomposition_defaults,

        // UI State
        ui: state.ui,
      }),
    }
  )
);

// ============================================================================
// Selectors
// ============================================================================

/**
 * Get current LLM configuration
 */
export const getLLMConfig = () => {
  const state = useConfigStore.getState();
  return {
    provider: state.provider,
    api_key: state.api_key,
    base_url: state.base_url,
    model_leanaide: state.model_leanaide,
    model_text: state.model_text,
    model_img: state.model_img,
    temperature: state.temperature,
    top_p: state.top_p,
    max_tokens: state.max_tokens,
    frequency_penalty: state.frequency_penalty,
    presence_penalty: state.presence_penalty,
  };
};

/**
 * Get UI state
 */
export const getUIState = () => {
  return useConfigStore.getState().ui;
};

/**
 * Check if API key is configured
 */
export const hasApiKey = () => {
  return !!useConfigStore.getState().api_key;
};

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook to get LLM config
 */
export const useLLMConfig = () => {
  return useConfigStore((state) => ({
    provider: state.provider,
    api_key: state.api_key,
    base_url: state.base_url,
    model_leanaide: state.model_leanaide,
    model_text: state.model_text,
    model_img: state.model_img,
    temperature: state.temperature,
    top_p: state.top_p,
    max_tokens: state.max_tokens,
    frequency_penalty: state.frequency_penalty,
    presence_penalty: state.presence_penalty,
    setLLMProvider: state.setLLMProvider,
    setApiKey: state.setApiKey,
    setBaseUrl: state.setBaseUrl,
    setModelLeanAide: state.setModelLeanAide,
    setModelText: state.setModelText,
    setModelImg: state.setModelImg,
    setTemperature: state.setTemperature,
    setTopP: state.setTopP,
    setMaxTokens: state.setMaxTokens,
    setFrequencyPenalty: state.setFrequencyPenalty,
    setPresencePenalty: state.setPresencePenalty,
  }));
};

/**
 * Hook to get UI state
 */
export const useUIState = () => {
  return useConfigStore((state) => state.ui);
};

/**
 * Hook to get ICR config
 */
export const useICRConfig = () => {
  return useConfigStore((state) => ({
    auto_refine_enabled: state.auto_refine_enabled,
    reward_calibration_enabled: state.reward_calibration_enabled,
    reward_calibration_threshold: state.reward_calibration_threshold,
    heatmap_analysis_enabled: state.heatmap_analysis_enabled,
    heatmap_snapshot_interval: state.heatmap_snapshot_interval,
    vlm_provider: state.vlm_provider,
    vlm_model: state.vlm_model,
    setAutoRefineEnabled: state.setAutoRefineEnabled,
    setRewardCalibrationEnabled: state.setRewardCalibrationEnabled,
    setRewardCalibrationThreshold: state.setRewardCalibrationThreshold,
    setHeatmapAnalysisEnabled: state.setHeatmapAnalysisEnabled,
    setHeatmapSnapshotInterval: state.setHeatmapSnapshotInterval,
    setVlmProvider: state.setVlmProvider,
    setVlmModel: state.setVlmModel,
  }));
};

/**
 * Hook to get defaults config
 */
export const useDefaultsConfig = () => {
  return useConfigStore((state) => ({
    determinism_defaults: state.determinism_defaults,
    decomposition_defaults: state.decomposition_defaults,
    setDeterminismDefaults: state.setDeterminismDefaults,
    setDecompositionDefaults: state.setDecompositionDefaults,
  }));
};

/**
 * Hook to get/set selected flow ID
 */
export const useSelectedFlowId = () => {
  return useConfigStore((state) => ({
    selectedFlowId: state.ui.selectedFlowId,
    setSelectedFlowId: state.setSelectedFlowId,
  }));
};
