// @ts-nocheck
import { useEffect, useMemo, useState } from 'react';
import { cn } from '@/lib/utils';
import { useConfig } from '@/services/hooks/useApi';
import { useSettingsStore } from '@/stores/settingsStore';
import { useWorkflowModels, useWorkflowConfig } from '@/services/hooks/useWorkflows';
import {
  BubbleBadge,
  BubbleButton,
  BubbleCard,
  BubbleField,
  BubbleInput,
  BubbleSelect,
  BubbleTextArea,
  BubbleToggle,
} from '@/components/bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

const fallbackProviders = [
  { provider: 'openai', name: 'OpenAI', models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'], requires_api_key: true },
  { provider: 'anthropic', name: 'Anthropic', models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'], requires_api_key: true },
  { provider: 'cohere', name: 'Cohere', models: ['command', 'command-light', 'command-nightly'], requires_api_key: true },
  { provider: 'local', name: 'Local', models: ['llama', 'mistral', 'custom'], requires_api_key: false },
];

function ProviderSettingsPanelBase({ className }: { className?: string }) {
  const {
    providers: providerResponse,
    isLoading,
    saveApiKey,
    updateParameters,
    isSavingKey,
    isUpdatingParams,
  } = useConfig();

  const {
    scope,
    provider,
    model,
    parameterSettings,
    setScope,
    setProvider,
    setModel,
    updateGlobalSettings,
    updateProviderSettings,
    updateModelSettings,
    setProviderApiKeyLastFour,
  } = useSettingsStore();

  const { config, updateConfig } = useWorkflowConfig();
  const { models, updateModels, add: addModel, remove: removeModel } = useWorkflowModels();

  const providers = providerResponse?.length ? providerResponse : fallbackProviders;

  const selectedProvider = provider || providers[0]?.provider || '';
  const selectedProviderConfig = providers.find((item) => item.provider === selectedProvider);
  const modelOptions = selectedProviderConfig?.models ?? [];
  const selectedModel = model || modelOptions[0] || '';

  useEffect(() => {
    if (!provider && providers.length) {
      setProvider(providers[0].provider);
    }
  }, [provider, providers, setProvider]);

  useEffect(() => {
    if (scope === 'Model' && selectedProvider && !model && modelOptions.length) {
      setModel(modelOptions[0]);
    }
  }, [scope, selectedProvider, model, modelOptions, setModel]);

  const scopedSettings = useMemo(() => {
    const globalSettings = parameterSettings.global;
    if (!selectedProvider) {
      return globalSettings;
    }
    const providerSettings =
      parameterSettings.providers[selectedProvider]?.settings || globalSettings;
    if (scope !== 'Model' || !selectedModel) {
      return scope === 'Provider' ? providerSettings : globalSettings;
    }
    return (
      parameterSettings.providers[selectedProvider]?.models?.[selectedModel] ||
      providerSettings
    );
  }, [parameterSettings, scope, selectedProvider, selectedModel]);

  const handleScopedUpdate = (updates: {
    generation?: Partial<typeof scopedSettings.generation>;
    evolution?: Partial<typeof scopedSettings.evolution>;
  }) => {
    if (scope === 'Global') {
      updateGlobalSettings(updates);
      if (updates.generation?.temperature !== undefined) {
        updateConfig({ temperature: updates.generation.temperature });
      }
      if (updates.generation?.top_p !== undefined) {
        updateConfig({ top_p: updates.generation.top_p });
      }
      if (updates.evolution?.max_iterations !== undefined) {
        updateConfig({ max_iterations: updates.evolution.max_iterations });
      }
      if (updates.evolution?.population_size !== undefined) {
        updateConfig({ population_size: updates.evolution.population_size });
      }
      return;
    }

    if (!selectedProvider) {
      return;
    }

    if (scope === 'Provider') {
      updateProviderSettings(selectedProvider, updates);
      return;
    }

    if (scope === 'Model' && selectedModel) {
      updateModelSettings(selectedProvider, selectedModel, updates);
    }
  };

  const [apiKeyInput, setApiKeyInput] = useState('');

  const handleSaveApiKey = async () => {
    if (!selectedProvider || !apiKeyInput.trim()) {
      return;
    }
    const response = await saveApiKey({ provider: selectedProvider, apiKey: apiKeyInput });
    if (response?.api_key_last_four) {
      setProviderApiKeyLastFour(selectedProvider, response.api_key_last_four);
      setApiKeyInput('');
    }
  };

  const handleSaveParameters = async () => {
    if (scope !== 'Global') {
      return;
    }
    await updateParameters({
      generation: {
        temperature: scopedSettings.generation.temperature,
        top_p: scopedSettings.generation.top_p,
        max_tokens: scopedSettings.generation.max_tokens,
      },
      evolution: {
        max_iterations: scopedSettings.evolution.max_iterations,
        population_size: scopedSettings.evolution.population_size,
      },
    });
  };

  const handleFeatureDimensions = (value: string) => {
    const dimensions = value
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean);
    handleScopedUpdate({ evolution: { feature_dimensions: dimensions } });
  };

  const handleModelUpdate = (index: number, updates: Partial<(typeof models)[number]>) => {
    const next = models.map((item, idx) => (idx === index ? { ...item, ...updates } : item));
    updateModels(next);
  };

  const addModelRow = () => {
    const providerName = selectedProvider || providers[0]?.provider || 'openai';
    const availableModels = providers.find((item) => item.provider === providerName)?.models;
    addModel({
      provider: providerName,
      model: availableModels?.[0] || 'gpt-4',
    });
  };

  return (
    <aside className={cn('w-96 bg-slate-50 border-r border-slate-200 overflow-y-auto', className)}>
      <div className="p-6 space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Configuration</h2>
          <p className="text-xs text-slate-500">
            Manage providers, API keys, and evolution defaults.
          </p>
        </div>

        <BubbleCard title="Settings Scope" description="Apply changes globally, per provider, or per model.">
          <div className="grid grid-cols-3 gap-2">
            {(['Global', 'Provider', 'Model'] as const).map((item) => (
              <BubbleButton
                key={item}
                variant={scope === item ? 'primary' : 'secondary'}
                onClick={() => setScope(item)}
              >
                {item}
              </BubbleButton>
            ))}
          </div>
        </BubbleCard>

        <BubbleCard title="Provider" description="Select provider and model defaults.">
          <div className="space-y-4">
            <BubbleField label="Provider">
              <BubbleSelect
                value={selectedProvider}
                onChange={(event) => setProvider(event.target.value)}
              >
                {providers.map((item) => (
                  <option key={item.provider} value={item.provider}>
                    {item.name || item.provider}
                  </option>
                ))}
              </BubbleSelect>
            </BubbleField>

            <BubbleField label="Model" hint={scope === 'Model' ? 'Scoped model settings' : 'Default model selection'}>
              <BubbleSelect
                value={selectedModel}
                onChange={(event) => setModel(event.target.value)}
                disabled={scope !== 'Model'}
              >
                {modelOptions.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </BubbleSelect>
            </BubbleField>

            <div className="rounded-lg border border-dashed border-slate-200 bg-white px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  API Key
                </span>
                {selectedProviderConfig?.requires_api_key ? (
                  <BubbleBadge tone="warning">Required</BubbleBadge>
                ) : (
                  <BubbleBadge tone="neutral">Optional</BubbleBadge>
                )}
              </div>
              <div className="mt-3 space-y-2">
                <BubbleInput
                  type="password"
                  placeholder="Enter API key"
                  value={apiKeyInput}
                  onChange={(event) => setApiKeyInput(event.target.value)}
                />
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>
                    {selectedProvider
                      ? `Saved key: ${parameterSettings.providers[selectedProvider]?.apiKeyLastFour ? `****${parameterSettings.providers[selectedProvider]?.apiKeyLastFour}` : 'none'}`
                      : 'Select a provider'}
                  </span>
                  <BubbleButton
                    variant="secondary"
                    onClick={handleSaveApiKey}
                    disabled={isSavingKey || !apiKeyInput.trim()}
                  >
                    {isSavingKey ? 'Saving...' : 'Save'}
                  </BubbleButton>
                </div>
              </div>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard title="Evolution Mode" description="Tune mode and model ensemble.">
          <div className="space-y-4">
            <BubbleField label="Mode">
              <BubbleSelect
                value={config.mode}
                onChange={(event) => updateConfig({ mode: event.target.value as typeof config.mode })}
              >
                <option value="standard">Standard</option>
                <option value="quality_diversity">Quality + Diversity</option>
                <option value="island_model">Island Model</option>
              </BubbleSelect>
            </BubbleField>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Model Ensemble
                </span>
                <BubbleButton variant="secondary" onClick={addModelRow}>
                  Add
                </BubbleButton>
              </div>
              {models.length === 0 && (
                <p className="text-xs text-slate-400">No models configured yet.</p>
              )}
              {models.map((item, index) => (
                <div key={`${item.provider}-${item.model}-${index}`} className="grid grid-cols-6 gap-2">
                  <BubbleSelect
                    className="col-span-2"
                    value={item.provider}
                    onChange={(event) => handleModelUpdate(index, { provider: event.target.value, model: '' })}
                  >
                    {providers.map((providerItem) => (
                      <option key={providerItem.provider} value={providerItem.provider}>
                        {providerItem.name || providerItem.provider}
                      </option>
                    ))}
                  </BubbleSelect>
                  <BubbleSelect
                    className="col-span-3"
                    value={item.model}
                    onChange={(event) => handleModelUpdate(index, { model: event.target.value })}
                  >
                    {(providers.find((providerItem) => providerItem.provider === item.provider)?.models ||
                      modelOptions
                    ).map((modelName) => (
                      <option key={modelName} value={modelName}>
                        {modelName}
                      </option>
                    ))}
                  </BubbleSelect>
                  <BubbleButton
                    className="col-span-1"
                    variant="ghost"
                    onClick={() => removeModel(index)}
                  >
                    Remove
                  </BubbleButton>
                </div>
              ))}
            </div>
          </div>
        </BubbleCard>

        <BubbleCard title="Generation Settings" description="Adjust sampling and API settings.">
          <div className="space-y-4">
            <BubbleField label="Temperature">
              <BubbleInput
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={scopedSettings.generation.temperature}
                onChange={(event) =>
                  handleScopedUpdate({
                    generation: { temperature: Number(event.target.value) },
                  })
                }
              />
            </BubbleField>
            <BubbleField label="Top P">
              <BubbleInput
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={scopedSettings.generation.top_p}
                onChange={(event) =>
                  handleScopedUpdate({ generation: { top_p: Number(event.target.value) } })
                }
              />
            </BubbleField>
            <BubbleField label="Max Tokens">
              <BubbleInput
                type="number"
                min="1"
                value={scopedSettings.generation.max_tokens}
                onChange={(event) =>
                  handleScopedUpdate({
                    generation: { max_tokens: Number(event.target.value) },
                  })
                }
              />
            </BubbleField>
            <div className="grid grid-cols-2 gap-3">
              <BubbleField label="Frequency Penalty">
                <BubbleInput
                  type="number"
                  step="0.1"
                  value={scopedSettings.generation.frequency_penalty}
                  onChange={(event) =>
                    handleScopedUpdate({
                      generation: { frequency_penalty: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
              <BubbleField label="Presence Penalty">
                <BubbleInput
                  type="number"
                  step="0.1"
                  value={scopedSettings.generation.presence_penalty}
                  onChange={(event) =>
                    handleScopedUpdate({
                      generation: { presence_penalty: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <BubbleField label="API Timeout (s)">
                <BubbleInput
                  type="number"
                  min="5"
                  value={scopedSettings.generation.api_timeout || 30}
                  onChange={(event) =>
                    handleScopedUpdate({
                      generation: { api_timeout: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
              <BubbleField label="API Retries">
                <BubbleInput
                  type="number"
                  min="0"
                  value={scopedSettings.generation.api_retries || 0}
                  onChange={(event) =>
                    handleScopedUpdate({
                      generation: { api_retries: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
            </div>
            <BubbleField label="Reasoning Effort">
              <BubbleSelect
                value={scopedSettings.generation.reasoning_effort || 'medium'}
                onChange={(event) =>
                  handleScopedUpdate({
                    generation: { reasoning_effort: event.target.value as 'low' | 'medium' | 'high' },
                  })
                }
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </BubbleSelect>
            </BubbleField>
          </div>
        </BubbleCard>

        <BubbleCard title="Evolution Settings" description="Control population and exploration.">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <BubbleField label="Max Iterations">
                <BubbleInput
                  type="number"
                  min="1"
                  value={scopedSettings.evolution.max_iterations}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { max_iterations: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
              <BubbleField label="Population Size">
                <BubbleInput
                  type="number"
                  min="1"
                  value={scopedSettings.evolution.population_size}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { population_size: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <BubbleField label="Islands">
                <BubbleInput
                  type="number"
                  min="1"
                  value={scopedSettings.evolution.num_islands || 1}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { num_islands: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
              <BubbleField label="Migration Interval">
                <BubbleInput
                  type="number"
                  min="1"
                  value={scopedSettings.evolution.migration_interval || 10}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { migration_interval: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <BubbleField label="Migration Rate">
                <BubbleInput
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={scopedSettings.evolution.migration_rate || 0.1}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { migration_rate: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
              <BubbleField label="Archive Size">
                <BubbleInput
                  type="number"
                  min="0"
                  value={scopedSettings.evolution.archive_size || 0}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { archive_size: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <BubbleField label="Elite Ratio">
                <BubbleInput
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={scopedSettings.evolution.elite_ratio || 0.1}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { elite_ratio: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
              <BubbleField label="Checkpoint Interval">
                <BubbleInput
                  type="number"
                  min="1"
                  value={scopedSettings.evolution.checkpoint_interval || 100}
                  onChange={(event) =>
                    handleScopedUpdate({
                      evolution: { checkpoint_interval: Number(event.target.value) },
                    })
                  }
                />
              </BubbleField>
            </div>
            <BubbleField label="Feature Dimensions" hint="Comma-separated feature tags for QD runs.">
              <BubbleTextArea
                rows={2}
                value={(scopedSettings.evolution.feature_dimensions || []).join(', ')}
                onChange={(event) => handleFeatureDimensions(event.target.value)}
              />
            </BubbleField>
            <BubbleField label="Diversity Metric">
              <BubbleSelect
                value={scopedSettings.evolution.diversity_metric || 'edit_distance'}
                onChange={(event) =>
                  handleScopedUpdate({
                    evolution: { diversity_metric: event.target.value },
                  })
                }
              >
                <option value="edit_distance">Edit Distance</option>
                <option value="semantic">Semantic</option>
                <option value="structure">Structure</option>
              </BubbleSelect>
            </BubbleField>
          </div>
        </BubbleCard>

        <BubbleCard title="Apply Settings" description="Save global defaults to the backend.">
          <div className="flex items-center justify-between">
            <BubbleToggle
              checked={scope === 'Global'}
              onChange={() => setScope('Global')}
              label="Use Global"
            />
            <BubbleButton
              onClick={handleSaveParameters}
              disabled={scope !== 'Global' || isUpdatingParams}
            >
              {isUpdatingParams ? 'Saving...' : 'Save Defaults'}
            </BubbleButton>
          </div>
          {isLoading && <p className="mt-2 text-xs text-slate-400">Loading configuration...</p>}
        </BubbleCard>
      </div>
    </aside>
  );
}

export const ProviderSettingsPanel = withComponentBoundary(
  ProviderSettingsPanelBase,
  'ProviderSettingsPanel'
);
