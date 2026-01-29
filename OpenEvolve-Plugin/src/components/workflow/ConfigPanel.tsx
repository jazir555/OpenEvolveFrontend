import { useWorkflowConfig, useWorkflowModels } from '@/services/hooks/useWorkflows';
import { cn } from '@/lib/utils';
import { useState } from 'react';
import { BubbleButton, BubbleField, BubbleInput, BubbleSelect } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

function ConfigPanelBase({ className }: { className?: string }) {
  const { config, updateConfig, reset } = useWorkflowConfig();
  const { models, updateModels, add: addModel, remove: removeModel } = useWorkflowModels();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const providers = ['openai', 'anthropic', 'cohere', 'huggingface', 'local'];
  const modelsByProvider: Record<string, string[]> = {
    openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
    anthropic: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
    cohere: ['command', 'command-light', 'command-nightly'],
    huggingface: ['custom'],
    local: ['llama', 'mistral', 'custom'],
  };

  return (
    <aside className={cn('w-80 bg-white border-r border-gray-200 overflow-y-auto', className)}>
      <div className="p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>

        <div className="mb-6">
          <BubbleField label="Provider">
            <BubbleSelect
              value={config.mode}
              onChange={(e) => updateConfig({ mode: e.target.value as any })}
            >
              <option value="standard">Standard Mode</option>
              <option value="quality_diversity">Quality + Diversity</option>
              <option value="island_model">Island Model</option>
            </BubbleSelect>
          </BubbleField>
        </div>

        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Models</h3>
          {models.map((model, index) => (
            <div key={index} className="flex gap-2 mb-2">
              <BubbleSelect
                value={model.provider}
                onChange={(e) => {
                  const newModels = [...models];
                  newModels[index] = { ...model, provider: e.target.value };
                  updateModels(newModels);
                }}
                className="flex-1 text-sm"
              >
                {providers.map((provider) => (
                  <option key={provider} value={provider}>
                    {provider.charAt(0).toUpperCase() + provider.slice(1)}
                  </option>
                ))}
              </BubbleSelect>
              <BubbleButton
                onClick={() => removeModel(index)}
                variant="ghost"
                className="px-3 py-2 text-red-600"
              >
                Remove
              </BubbleButton>
            </div>
          ))}
          <BubbleButton
            onClick={() => addModel({ provider: 'openai', model: 'gpt-4' })}
            variant="secondary"
            className="w-full text-sm"
          >
            Add model
          </BubbleButton>
        </div>

        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-700">Parameters</h3>
            <BubbleButton
              onClick={() => setShowAdvanced(!showAdvanced)}
              variant="ghost"
              className="px-2 py-1 text-xs"
            >
              {showAdvanced ? 'Less' : 'Advanced'}
            </BubbleButton>
          </div>

          <div className="space-y-4">
            <BubbleField label="Max Iterations">
              <BubbleInput
                type="number"
                value={config.max_iterations}
                onChange={(e) => updateConfig({ max_iterations: parseInt(e.target.value) })}
              />
            </BubbleField>

            <BubbleField label="Population Size">
              <BubbleInput
                type="number"
                value={config.population_size}
                onChange={(e) => updateConfig({ population_size: parseInt(e.target.value) })}
              />
            </BubbleField>

            {showAdvanced && (
              <>
                <BubbleField label="Temperature">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={config.temperature}
                    onChange={(e) => updateConfig({ temperature: parseFloat(e.target.value) })}
                  />
                </BubbleField>

                <BubbleField label="Top P">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={config.top_p}
                    onChange={(e) => updateConfig({ top_p: parseFloat(e.target.value) })}
                  />
                </BubbleField>

                <BubbleField label="Mutation Rate">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={config.mutation_rate}
                    onChange={(e) => updateConfig({ mutation_rate: parseFloat(e.target.value) })}
                  />
                </BubbleField>

                <BubbleField label="Crossover Rate">
                  <BubbleInput
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={config.crossover_rate}
                    onChange={(e) => updateConfig({ crossover_rate: parseFloat(e.target.value) })}
                  />
                </BubbleField>
              </>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <BubbleButton onClick={reset} variant="secondary" className="w-full text-sm">
            Reset to Defaults
          </BubbleButton>
        </div>
      </div>
    </aside>
  );
}

export const ConfigPanel = withComponentBoundary(ConfigPanelBase, 'ConfigPanel');
