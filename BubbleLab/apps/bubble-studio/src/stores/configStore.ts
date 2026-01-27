/**
 * Configuration Store
 * Manages LLM configuration and application settings
 * Replaces Streamlit session state for configuration
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { LLMProvider, LLMConfig, UIState, PanelMode } from '../types/api';

// ============================================================================
// Configuration State Interface
// ============================================================================

interface ConfigState extends LLMConfig {
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

  // UI Actions
  setSelectedFlowId: (flowId: string | null) => void;
  setPanelMode: (mode: UIState['panelMode']) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setDarkMode: (darkMode: boolean) => void;
  setFontSize: (size: UIState['fontSize']) => void;
  setAutoSave: (autoSave: boolean) => void;

  // Bulk Actions
  setLLMConfig: (config: Partial<LLMConfig>) => void;
  setUIState: (state: Partial<UIState>) => void;

  // Reset
  resetLLMConfig: () => void;
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

// ============================================================================
// Create Store
// ============================================================================

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      // Initial State
      ...defaultLLMConfig,
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
 * Hook to get/set selected flow ID
 */
export const useSelectedFlowId = () => {
  return useConfigStore((state) => ({
    selectedFlowId: state.ui.selectedFlowId,
    setSelectedFlowId: state.setSelectedFlowId,
  }));
};
