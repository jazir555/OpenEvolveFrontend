/**
 * Settings Panel Component
 * Application and LLM configuration settings
 */

import { useEffect, useState } from 'react';
import { useConfigStore, useLLMConfig, useICRConfig, useDefaultsConfig } from '../../stores/configStore';
import { LLMProvider } from '../../types/api';
import apiClient from '../../lib/api-client';
import { notify } from '../common/Notifications';

export function SettingsPanel() {
  const {
    provider,
    api_key,
    base_url,
    model_leanaide,
    model_text,
    model_img,
    temperature,
    top_p,
    max_tokens,
    frequency_penalty,
    presence_penalty,
    setLLMProvider,
    setApiKey,
    setBaseUrl,
    setModelLeanAide,
    setModelText,
    setModelImg,
    setTemperature,
    setTopP,
    setMaxTokens,
    setFrequencyPenalty,
    setPresencePenalty,
  } = useLLMConfig();

  const {
    auto_refine_enabled,
    reward_calibration_enabled,
    reward_calibration_threshold,
    heatmap_analysis_enabled,
    heatmap_snapshot_interval,
    vlm_provider,
    vlm_model,
    setAutoRefineEnabled,
    setRewardCalibrationEnabled,
    setRewardCalibrationThreshold,
    setHeatmapAnalysisEnabled,
    setHeatmapSnapshotInterval,
    setVlmProvider,
    setVlmModel,
  } = useICRConfig();

  const [showApiKey, setShowApiKey] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoadingIcr, setIsLoadingIcr] = useState(false);
  const [isSavingIcr, setIsSavingIcr] = useState(false);
  const [isLoadingDefaults, setIsLoadingDefaults] = useState(false);
  const [isSavingDeterminismDefaults, setIsSavingDeterminismDefaults] = useState(false);
  const [isSavingDecompositionDefaults, setIsSavingDecompositionDefaults] = useState(false);

  const {
    determinism_defaults,
    decomposition_defaults,
    setDeterminismDefaults,
    setDecompositionDefaults,
  } = useDefaultsConfig();

  const [determinismConfigJson, setDeterminismConfigJson] = useState('{}');
  const [decompositionMakerConfigJson, setDecompositionMakerConfigJson] = useState('{}');
  const [decompositionClientConfigJson, setDecompositionClientConfigJson] = useState('{}');
  const [decompositionMdapConfigJson, setDecompositionMdapConfigJson] = useState('{}');

  const { ui, setAutoSave, setDarkMode } = useConfigStore((state) => ({
    ui: state.ui,
    setAutoSave: state.setAutoSave,
    setDarkMode: state.setDarkMode,
  }));

  useEffect(() => {
    let isMounted = true;

    const loadConfig = async () => {
      setIsLoading(true);
      try {
        const config = await apiClient.getLLMConfig();
        if (isMounted) {
          useConfigStore.getState().setLLMConfig(config);
        }
      } catch (error) {
        notify({
          type: 'error',
          title: 'Settings load failed',
          message:
            error instanceof Error
              ? error.message
              : 'Unable to load LLM configuration.',
        });
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    const loadIcrConfig = async () => {
      setIsLoadingIcr(true);
      try {
        const config = await apiClient.getICRConfig();
        if (isMounted) {
          const normalized = {
            ...config,
            vlm_provider: config.vlm_provider ?? '',
            vlm_model: config.vlm_model ?? '',
          };
          useConfigStore.getState().setICRConfig(normalized);
        }
      } catch (error) {
        notify({
          type: 'error',
          title: 'ICR settings load failed',
          message:
            error instanceof Error
              ? error.message
              : 'Unable to load ICR configuration.',
        });
      } finally {
        if (isMounted) setIsLoadingIcr(false);
      }
    };

    void loadConfig();
    void loadIcrConfig();
    const loadDefaults = async () => {
      setIsLoadingDefaults(true);
      try {
        const [determinism, decomposition] = await Promise.all([
          apiClient.getDeterminismDefaults(),
          apiClient.getDecompositionDefaults(),
        ]);
        if (isMounted) {
          setDeterminismDefaults(determinism);
          setDecompositionDefaults(decomposition);
        }
      } catch (error) {
        notify({
          type: 'error',
          title: 'Defaults load failed',
          message:
            error instanceof Error
              ? error.message
              : 'Unable to load integration defaults.',
        });
      } finally {
        if (isMounted) setIsLoadingDefaults(false);
      }
    };

    void loadDefaults();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    setDeterminismConfigJson(
      JSON.stringify(determinism_defaults.config || {}, null, 2)
    );
  }, [determinism_defaults.config]);

  useEffect(() => {
    setDecompositionMakerConfigJson(
      JSON.stringify(decomposition_defaults.maker_config || {}, null, 2)
    );
  }, [decomposition_defaults.maker_config]);

  useEffect(() => {
    setDecompositionClientConfigJson(
      JSON.stringify(decomposition_defaults.openevolve_client_config || {}, null, 2)
    );
  }, [decomposition_defaults.openevolve_client_config]);

  useEffect(() => {
    setDecompositionMdapConfigJson(
      JSON.stringify(decomposition_defaults.mdap_config || {}, null, 2)
    );
  }, [decomposition_defaults.mdap_config]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const updated = await apiClient.updateLLMConfig({
        provider,
        api_key,
        base_url,
        model_leanaide,
        model_text,
        model_img,
        temperature,
        top_p,
        max_tokens,
        frequency_penalty,
        presence_penalty,
      });
      useConfigStore.getState().setLLMConfig(updated);
      notify({
        type: 'success',
        title: 'Settings saved',
        message: 'LLM configuration updated successfully.',
      });
    } catch (error) {
      notify({
        type: 'error',
        title: 'Save failed',
        message:
          error instanceof Error
            ? error.message
            : 'Unable to save LLM configuration.',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveIcr = async () => {
    setIsSavingIcr(true);
    try {
      const cleanedProvider =
        vlm_provider && vlm_provider.trim().length > 0
          ? vlm_provider.trim()
          : undefined;
      const cleanedModel =
        vlm_model && vlm_model.trim().length > 0 ? vlm_model.trim() : undefined;

      const updated = await apiClient.updateICRConfig({
        auto_refine_enabled,
        reward_calibration_enabled,
        reward_calibration_threshold,
        heatmap_analysis_enabled,
        heatmap_snapshot_interval,
        vlm_provider: cleanedProvider,
        vlm_model: cleanedModel,
      });
      useConfigStore.getState().setICRConfig(updated);
      notify({
        type: 'success',
        title: 'ICR settings saved',
        message: 'ICR configuration updated successfully.',
      });
    } catch (error) {
      notify({
        type: 'error',
        title: 'ICR save failed',
        message:
          error instanceof Error
            ? error.message
            : 'Unable to save ICR configuration.',
      });
    } finally {
      setIsSavingIcr(false);
    }
  };

  const parseJsonField = (value: string, label: string) => {
    if (!value || !value.trim()) {
      return {};
    }
    try {
      return JSON.parse(value);
    } catch (error) {
      notify({
        type: 'error',
        title: 'Invalid JSON',
        message: `Failed to parse ${label}.`,
      });
      throw error;
    }
  };

  const normalizeOptional = (value: string | undefined) => {
    if (!value) return undefined;
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  };

  const updateDeterminismField = (field: keyof typeof determinism_defaults, value: any) => {
    setDeterminismDefaults({
      ...determinism_defaults,
      [field]: value,
    });
  };

  const updateDecompositionField = (field: keyof typeof decomposition_defaults, value: any) => {
    setDecompositionDefaults({
      ...decomposition_defaults,
      [field]: value,
    });
  };

  const handleSaveDeterminismDefaults = async () => {
    setIsSavingDeterminismDefaults(true);
    try {
      const configObject = parseJsonField(determinismConfigJson, 'determinism config JSON');
      const updated = await apiClient.updateDeterminismDefaults({
        ...determinism_defaults,
        mode: normalizeOptional(determinism_defaults.mode) || 'auto',
        cloud_provider: normalizeOptional(determinism_defaults.cloud_provider),
        cloud_model: normalizeOptional(determinism_defaults.cloud_model),
        cloud_base_url: normalizeOptional(determinism_defaults.cloud_base_url),
        local_provider: normalizeOptional(determinism_defaults.local_provider) || 'hf',
        local_model: normalizeOptional(determinism_defaults.local_model),
        local_device: normalizeOptional(determinism_defaults.local_device) || 'cpu',
        local_dtype: normalizeOptional(determinism_defaults.local_dtype) || 'auto',
        detllm_backend: normalizeOptional(determinism_defaults.detllm_backend),
        detllm_model: normalizeOptional(determinism_defaults.detllm_model),
        check_provider: normalizeOptional(determinism_defaults.check_provider),
        check_model: normalizeOptional(determinism_defaults.check_model),
        check_base_url: normalizeOptional(determinism_defaults.check_base_url),
        check_device: normalizeOptional(determinism_defaults.check_device) || 'cpu',
        check_dtype: normalizeOptional(determinism_defaults.check_dtype) || 'auto',
        config: configObject,
      });
      setDeterminismDefaults(updated);
      notify({
        type: 'success',
        title: 'Determinism defaults saved',
        message: 'Determinism defaults updated successfully.',
      });
    } catch {
      // parseJsonField already notifies
    } finally {
      setIsSavingDeterminismDefaults(false);
    }
  };

  const handleSaveDecompositionDefaults = async () => {
    setIsSavingDecompositionDefaults(true);
    try {
      const makerConfig = parseJsonField(decompositionMakerConfigJson, 'maker config JSON');
      const clientConfig = parseJsonField(decompositionClientConfigJson, 'OpenEvolve client config JSON');
      const mdapConfig = parseJsonField(decompositionMdapConfigJson, 'MDAP config JSON');

      const updated = await apiClient.updateDecompositionDefaults({
        ...decomposition_defaults,
        strategy: normalizeOptional(decomposition_defaults.strategy),
        maker_config: makerConfig,
        openevolve_client_config: clientConfig,
        mdap_config: mdapConfig,
      });
      setDecompositionDefaults(updated);
      notify({
        type: 'success',
        title: 'Decomposition defaults saved',
        message: 'Decomposition defaults updated successfully.',
      });
    } catch {
      // parseJsonField already notifies
    } finally {
      setIsSavingDecompositionDefaults(false);
    }
  };

  const toggleDarkMode = () => {
    const nextValue = !ui.darkMode;
    setDarkMode(nextValue);
    if (nextValue) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  return (
    <div className="space-y-6">
      {/* LLM Provider Settings */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          LLM Configuration
        </h2>

        <div className="space-y-4">
          {/* Provider Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Provider
            </label>
            <select
              value={provider}
              onChange={(e) => setLLMProvider(e.target.value as LLMProvider)}
              className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
              <option value={LLMProvider.OPENAI}>OpenAI</option>
              <option value={LLMProvider.ANTHROPIC}>Anthropic</option>
              <option value={LLMProvider.COHERE}>Cohere</option>
              <option value={LLMProvider.CUSTOM}>Custom</option>
            </select>
          </div>

          {/* API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              API Key
            </label>
            <div className="mt-1 flex rounded-md shadow-sm">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={api_key}
                onChange={(e) => setApiKey(e.target.value)}
                className="block w-full rounded-l-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="sk-..."
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="inline-flex items-center rounded-r-lg border border-l-0 border-gray-300 bg-gray-50 px-3 text-sm text-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
              >
                {showApiKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          {/* Base URL (optional) */}
          {provider === LLMProvider.CUSTOM && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Base URL
              </label>
              <input
                type="text"
                value={base_url || ''}
                onChange={(e) => setBaseUrl(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="https://api.example.com"
              />
            </div>
          )}

          {/* Models */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                LeanAide Model
              </label>
              <input
                type="text"
                value={model_leanaide}
                onChange={(e) => setModelLeanAide(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="gpt-4"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Text Model
              </label>
              <input
                type="text"
                value={model_text}
                onChange={(e) => setModelText(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="gpt-3.5-turbo"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Image Model
              </label>
              <input
                type="text"
                value={model_img}
                onChange={(e) => setModelImg(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="gpt-4-vision-preview"
              />
            </div>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Temperature: {temperature}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="mt-1 block w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Top P: {top_p}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={top_p}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="mt-1 block w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Max Tokens
              </label>
              <input
                type="number"
                min="1"
                value={max_tokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Frequency Penalty: {frequency_penalty}
              </label>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={frequency_penalty}
                onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                className="mt-1 block w-full"
              />
            </div>
          </div>

          {/* Save Button */}
          <div className="pt-4">
            <button
              onClick={handleSave}
              disabled={isSaving || isLoading}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSaving ? 'Saving...' : isLoading ? 'Loading...' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>

      {/* UI Preferences */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          UI Preferences
        </h2>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Auto-save
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Automatically save your work
              </p>
            </div>
            <button
              onClick={() => setAutoSave(!ui.autoSave)}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${ui.autoSave ? 'bg-blue-600' : 'bg-gray-200'}`}
              role="switch"
              aria-checked={ui.autoSave}
            >
              <span
                aria-hidden="true"
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${ui.autoSave ? 'translate-x-5' : 'translate-x-0'}`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Dark Mode
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Use dark theme
              </p>
            </div>
            <button
              onClick={toggleDarkMode}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${ui.darkMode ? 'bg-blue-600' : 'bg-gray-200'}`}
              role="switch"
              aria-checked={ui.darkMode}
            >
              <span
                aria-hidden="true"
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${ui.darkMode ? 'translate-x-5' : 'translate-x-0'}`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Integration Defaults */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Integration Defaults
        </h2>

        <div className="space-y-8">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Determinism Defaults
            </h3>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Mode
                </label>
                <select
                  value={determinism_defaults.mode}
                  onChange={(e) => updateDeterminismField('mode', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                >
                  <option value="auto">Auto</option>
                  <option value="cloud">Cloud</option>
                  <option value="local">Local</option>
                  <option value="hybrid">Hybrid</option>
                  <option value="consensus">Consensus</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Cloud Provider
                </label>
                <input
                  type="text"
                  value={determinism_defaults.cloud_provider || ''}
                  onChange={(e) => updateDeterminismField('cloud_provider', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="openai"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Cloud Model
                </label>
                <input
                  type="text"
                  value={determinism_defaults.cloud_model || ''}
                  onChange={(e) => updateDeterminismField('cloud_model', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="gpt-4o"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Cloud Base URL
                </label>
                <input
                  type="text"
                  value={determinism_defaults.cloud_base_url || ''}
                  onChange={(e) => updateDeterminismField('cloud_base_url', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="https://api.openai.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Local Provider
                </label>
                <input
                  type="text"
                  value={determinism_defaults.local_provider || ''}
                  onChange={(e) => updateDeterminismField('local_provider', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="hf"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Local Model
                </label>
                <input
                  type="text"
                  value={determinism_defaults.local_model || ''}
                  onChange={(e) => updateDeterminismField('local_model', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="llama-3-8b"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Local Device
                </label>
                <input
                  type="text"
                  value={determinism_defaults.local_device || ''}
                  onChange={(e) => updateDeterminismField('local_device', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="cpu"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Local DType
                </label>
                <input
                  type="text"
                  value={determinism_defaults.local_dtype || ''}
                  onChange={(e) => updateDeterminismField('local_dtype', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="auto"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  DetLLM Backend
                </label>
                <input
                  type="text"
                  value={determinism_defaults.detllm_backend || ''}
                  onChange={(e) => updateDeterminismField('detllm_backend', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="torch"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  DetLLM Model
                </label>
                <input
                  type="text"
                  value={determinism_defaults.detllm_model || ''}
                  onChange={(e) => updateDeterminismField('detllm_model', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="detllm-v1"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Check Tier
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  value={determinism_defaults.check_tier}
                  onChange={(e) => {
                    const nextValue = parseInt(e.target.value, 10);
                    updateDeterminismField(
                      'check_tier',
                      Number.isFinite(nextValue) ? nextValue : 1
                    );
                  }}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Check Runs
                </label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={determinism_defaults.check_runs}
                  onChange={(e) => {
                    const nextValue = parseInt(e.target.value, 10);
                    updateDeterminismField(
                      'check_runs',
                      Number.isFinite(nextValue) ? nextValue : 1
                    );
                  }}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Determinism Config Overrides (JSON)
              </label>
              <textarea
                value={determinismConfigJson}
                onChange={(e) => setDeterminismConfigJson(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                rows={5}
              />
            </div>

            <div className="pt-2">
              <button
                onClick={handleSaveDeterminismDefaults}
                disabled={isSavingDeterminismDefaults || isLoadingDefaults}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isSavingDeterminismDefaults
                  ? 'Saving...'
                  : isLoadingDefaults
                    ? 'Loading...'
                    : 'Save Determinism Defaults'}
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Decomposition Defaults
            </h3>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Default Strategy
                </label>
                <input
                  type="text"
                  value={decomposition_defaults.strategy || ''}
                  onChange={(e) => updateDecompositionField('strategy', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  placeholder="hybrid"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    Adaptive Selection
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Use adaptive strategy selection in decomposition.
                  </p>
                </div>
                <button
                  onClick={() =>
                    updateDecompositionField('enable_adaptive_selection', !decomposition_defaults.enable_adaptive_selection)
                  }
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${decomposition_defaults.enable_adaptive_selection ? 'bg-blue-600' : 'bg-gray-200'}`}
                  role="switch"
                  aria-checked={decomposition_defaults.enable_adaptive_selection}
                >
                  <span
                    aria-hidden="true"
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${decomposition_defaults.enable_adaptive_selection ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Workflow Max Refinement Loops
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={decomposition_defaults.workflow_max_refinement_loops}
                  onChange={(e) => {
                    const nextValue = parseInt(e.target.value, 10);
                    updateDecompositionField(
                      'workflow_max_refinement_loops',
                      Number.isFinite(nextValue) ? nextValue : 1
                    );
                  }}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    MDAP Enabled
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Default MDAP toggle for decomposition workflows.
                  </p>
                </div>
                <button
                  onClick={() => updateDecompositionField('mdap_enabled', !decomposition_defaults.mdap_enabled)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${decomposition_defaults.mdap_enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
                  role="switch"
                  aria-checked={decomposition_defaults.mdap_enabled}
                >
                  <span
                    aria-hidden="true"
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${decomposition_defaults.mdap_enabled ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    MAKER Enabled
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Default MAKER toggle for decomposition workflows.
                  </p>
                </div>
                <button
                  onClick={() => updateDecompositionField('maker_enabled', !decomposition_defaults.maker_enabled)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${decomposition_defaults.maker_enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
                  role="switch"
                  aria-checked={decomposition_defaults.maker_enabled}
                >
                  <span
                    aria-hidden="true"
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${decomposition_defaults.maker_enabled ? 'translate-x-5' : 'translate-x-0'}`}
                  />
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Maker Config (JSON)
              </label>
              <textarea
                value={decompositionMakerConfigJson}
                onChange={(e) => setDecompositionMakerConfigJson(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                rows={4}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                MDAP Config (JSON)
              </label>
              <textarea
                value={decompositionMdapConfigJson}
                onChange={(e) => setDecompositionMdapConfigJson(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                rows={4}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                OpenEvolve Client Config (JSON)
              </label>
              <textarea
                value={decompositionClientConfigJson}
                onChange={(e) => setDecompositionClientConfigJson(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                rows={4}
              />
            </div>

            <div className="pt-2">
              <button
                onClick={handleSaveDecompositionDefaults}
                disabled={isSavingDecompositionDefaults || isLoadingDefaults}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isSavingDecompositionDefaults
                  ? 'Saving...'
                  : isLoadingDefaults
                    ? 'Loading...'
                    : 'Save Decomposition Defaults'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ICR Configuration */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          ICR Configuration
        </h2>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Auto-refine
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Automatically trigger refinement cycles on weak scores.
              </p>
            </div>
            <button
              onClick={() => setAutoRefineEnabled(!auto_refine_enabled)}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${auto_refine_enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
              role="switch"
              aria-checked={auto_refine_enabled}
            >
              <span
                aria-hidden="true"
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${auto_refine_enabled ? 'translate-x-5' : 'translate-x-0'}`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Reward calibration
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Ask for preference feedback when confidence is low.
              </p>
            </div>
            <button
              onClick={() =>
                setRewardCalibrationEnabled(!reward_calibration_enabled)
              }
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${reward_calibration_enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
              role="switch"
              aria-checked={reward_calibration_enabled}
            >
              <span
                aria-hidden="true"
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${reward_calibration_enabled ? 'translate-x-5' : 'translate-x-0'}`}
              />
            </button>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Reward calibration threshold
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={reward_calibration_threshold}
              onChange={(e) => {
                const nextValue = parseFloat(e.target.value);
                setRewardCalibrationThreshold(Number.isFinite(nextValue) ? nextValue : 0);
              }}
              className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Heatmap analysis
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Analyze interaction heatmaps for UI friction.
              </p>
            </div>
            <button
              onClick={() => setHeatmapAnalysisEnabled(!heatmap_analysis_enabled)}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${heatmap_analysis_enabled ? 'bg-blue-600' : 'bg-gray-200'}`}
              role="switch"
              aria-checked={heatmap_analysis_enabled}
            >
              <span
                aria-hidden="true"
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${heatmap_analysis_enabled ? 'translate-x-5' : 'translate-x-0'}`}
              />
            </button>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Heatmap snapshot interval
              </label>
              <input
                type="number"
                min="1"
              value={heatmap_snapshot_interval}
              onChange={(e) => {
                const nextValue = parseInt(e.target.value, 10);
                setHeatmapSnapshotInterval(Number.isFinite(nextValue) ? nextValue : 1);
              }}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                VLM Provider (optional)
              </label>
              <input
                type="text"
                value={vlm_provider || ''}
                onChange={(e) => setVlmProvider(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="openai"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              VLM Model (optional)
            </label>
            <input
              type="text"
              value={vlm_model || ''}
              onChange={(e) => setVlmModel(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              placeholder="gpt-4o"
            />
          </div>

          <div className="pt-2">
            <button
              onClick={handleSaveIcr}
              disabled={isSavingIcr || isLoadingIcr}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSavingIcr
                ? 'Saving...'
                : isLoadingIcr
                  ? 'Loading...'
                  : 'Save ICR Settings'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
