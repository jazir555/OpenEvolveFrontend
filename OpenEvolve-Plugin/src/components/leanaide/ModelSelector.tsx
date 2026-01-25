import { useState } from 'react';
import { cn } from '@/lib/utils';
import { BubbleBadge, BubbleButton, BubbleField, BubbleSelect } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface LeanModel {
  id: string;
  name: string;
  provider: string;
  description?: string;
  capabilities: string[];
}

interface ModelSelectorProps {
  models: LeanModel[];
  selectedModel: string;
  onModelChange: (modelId: string) => void;
  className?: string;
}

function ModelSelectorBase({
  models,
  selectedModel,
  onModelChange,
  className,
}: ModelSelectorProps) {
  const [showDetails, setShowDetails] = useState(false);

  const selectedModelData = models.find((m) => m.id === selectedModel);

  const groupedModels = models.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  }, {} as Record<string, LeanModel[]>);

  return (
    <div className={cn('model-selector', className)}>
      <div className="mb-3">
        <BubbleField label="Select Model">
          <BubbleSelect
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          >
            {Object.entries(groupedModels).map(([provider, providerModels]) => (
              <optgroup key={provider} label={provider.toUpperCase()}>
                {providerModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </BubbleSelect>
        </BubbleField>
      </div>

      {/* Model Details Toggle */}
      <BubbleButton
        onClick={() => setShowDetails(!showDetails)}
        variant="ghost"
        className="px-0"
      >
        {showDetails ? 'Hide' : 'Show'} model details
      </BubbleButton>

      {/* Model Details */}
      {showDetails && selectedModelData && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-2">
          <div>
            <div className="text-sm font-medium text-gray-900">{selectedModelData.name}</div>
            <div className="text-xs text-gray-600">{selectedModelData.provider}</div>
          </div>

          {selectedModelData.description && (
            <div className="text-sm text-gray-700">{selectedModelData.description}</div>
          )}

          {selectedModelData.capabilities.length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-700 mb-1">Capabilities</div>
              <div className="flex flex-wrap gap-1">
                {selectedModelData.capabilities.map((capability) => (
                  <BubbleBadge key={capability} tone="info">
                    {capability}
                  </BubbleBadge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Quick Model List */}
      {!showDetails && (
        <div className="mt-3 space-y-1">
          <div className="text-xs text-gray-600 mb-1">Available Models:</div>
          {models.map((model) => (
            <BubbleButton
              key={model.id}
              onClick={() => onModelChange(model.id)}
              variant={selectedModel === model.id ? 'secondary' : 'ghost'}
              className={cn('w-full text-left justify-start')}
            >
              <div className="font-medium">{model.name}</div>
              <div className="text-xs text-gray-500">{model.provider}</div>
            </BubbleButton>
          ))}
        </div>
      )}
    </div>
  );
}

export const ModelSelector = withComponentBoundary(ModelSelectorBase, 'ModelSelector');
