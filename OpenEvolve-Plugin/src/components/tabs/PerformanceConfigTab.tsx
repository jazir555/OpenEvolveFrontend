// @ts-nocheck
import React from 'react';
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
import {
  BubbleBadge,
  BubbleButton,
  BubbleCard,
  BubbleCheckbox,
  BubbleField,
  BubbleInput,
  BubbleSelect,
} from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

/**
 * Performance Configuration Tab
 * UI for configuring performance optimization settings
 */
const PerformanceConfigTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
  onValidate: () => void;
}> = ({ config, onConfigUpdate, onValidate }) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;

    onConfigUpdate({
      performanceConfig: {
        ...config.performanceConfig,
        [name]: type === 'checkbox' ? checked : value,
      },
    });
  };

  const handleCachingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;

    onConfigUpdate({
      performanceConfig: {
        ...config.performanceConfig,
        caching: {
          ...config.performanceConfig?.caching,
          [name]: type === 'checkbox' ? checked : value,
        },
      },
    });
  };

  const handleParallelProcessingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;

    onConfigUpdate({
      performanceConfig: {
        ...config.performanceConfig,
        parallel_processing: {
          ...config.performanceConfig?.parallel_processing,
          [name]: type === 'checkbox' ? checked : value,
        },
      },
    });
  };

  const handleMemoryManagementChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;

    onConfigUpdate({
      performanceConfig: {
        ...config.performanceConfig,
        memory_management: {
          ...config.performanceConfig?.memory_management,
          [name]: type === 'checkbox' ? checked : value,
        },
      },
    });
  };

  return (
    <div className="space-y-6">
      <BubbleCard
        title="Performance Configuration"
        description="Control optimization and runtime efficiency features."
        actions={
          <BubbleBadge tone={config.performanceConfig?.enabled ? 'success' : 'neutral'}>
            {config.performanceConfig?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <BubbleCheckbox
          name="enabled"
          checked={config.performanceConfig?.enabled || false}
          onChange={handleInputChange}
          label="Enable performance optimization"
        />
      </BubbleCard>

      <BubbleCard
        title="Caching Configuration"
        description="Configure cache strategy and limits."
        actions={
          <BubbleBadge tone={config.performanceConfig?.caching?.enabled ? 'success' : 'neutral'}>
            {config.performanceConfig?.caching?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.performanceConfig?.caching?.enabled || false}
            onChange={handleCachingChange}
            label="Enable caching"
          />

          {config.performanceConfig?.caching?.enabled && (
            <div className="grid gap-4 md:grid-cols-2">
              <BubbleField label="Cache Strategy">
                <BubbleSelect
                  name="strategy"
                  value={config.performanceConfig?.caching?.strategy || 'lru'}
                  onChange={handleCachingChange}
                >
                  <option value="lru">LRU (Least Recently Used)</option>
                  <option value="lfu">LFU (Least Frequently Used)</option>
                  <option value="fifo">FIFO (First In First Out)</option>
                  <option value="random">Random</option>
                </BubbleSelect>
              </BubbleField>

              <BubbleField label="Max Cache Size">
                <BubbleInput
                  type="number"
                  name="max_size"
                  min="1"
                  max="10000"
                  value={config.performanceConfig?.caching?.max_size || 1000}
                  onChange={handleCachingChange}
                />
              </BubbleField>

              <BubbleField label="Cache TTL (seconds)">
                <BubbleInput
                  type="number"
                  name="ttl"
                  min="0"
                  max="86400"
                  value={config.performanceConfig?.caching?.ttl || 3600}
                  onChange={handleCachingChange}
                />
              </BubbleField>

              <BubbleField label="Compression Algorithm">
                <BubbleSelect
                  name="compression"
                  value={config.performanceConfig?.caching?.compression || 'gzip'}
                  onChange={handleCachingChange}
                >
                  <option value="gzip">GZIP</option>
                  <option value="brotli">Brotli</option>
                  <option value="none">None</option>
                </BubbleSelect>
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="Parallel Processing Configuration"
        description="Tune concurrency and batching."
        actions={
          <BubbleBadge tone={config.performanceConfig?.parallel_processing?.enabled ? 'success' : 'neutral'}>
            {config.performanceConfig?.parallel_processing?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.performanceConfig?.parallel_processing?.enabled || false}
            onChange={handleParallelProcessingChange}
            label="Enable parallel processing"
          />

          {config.performanceConfig?.parallel_processing?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Max Workers">
                <BubbleInput
                  type="number"
                  name="max_workers"
                  min="1"
                  max="100"
                  value={config.performanceConfig?.parallel_processing?.max_workers || 4}
                  onChange={handleParallelProcessingChange}
                />
              </BubbleField>

              <BubbleField label="Worker Type">
                <BubbleSelect
                  name="worker_type"
                  value={config.performanceConfig?.parallel_processing?.worker_type || 'thread'}
                  onChange={handleParallelProcessingChange}
                >
                  <option value="thread">Thread</option>
                  <option value="process">Process</option>
                  <option value="cluster">Cluster</option>
                </BubbleSelect>
              </BubbleField>

              <BubbleField label="Batch Size">
                <BubbleInput
                  type="number"
                  name="batch_size"
                  min="1"
                  max="1000"
                  value={config.performanceConfig?.parallel_processing?.batch_size || 10}
                  onChange={handleParallelProcessingChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="Memory Management Configuration"
        description="Define limits and garbage collection cadence."
        actions={
          <BubbleBadge tone={config.performanceConfig?.memory_management?.enabled ? 'success' : 'neutral'}>
            {config.performanceConfig?.memory_management?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.performanceConfig?.memory_management?.enabled || false}
            onChange={handleMemoryManagementChange}
            label="Enable memory management"
          />

          {config.performanceConfig?.memory_management?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Max Memory (MB)">
                <BubbleInput
                  type="number"
                  name="max_memory_mb"
                  min="100"
                  max="100000"
                  value={config.performanceConfig?.memory_management?.max_memory_mb || 4096}
                  onChange={handleMemoryManagementChange}
                />
              </BubbleField>

              <BubbleField label="Memory Threshold (%)">
                <BubbleInput
                  type="number"
                  name="memory_threshold_percent"
                  min="50"
                  max="95"
                  value={(config.performanceConfig?.memory_management as any)?.memory_threshold_percent || 80}
                  onChange={handleMemoryManagementChange}
                />
              </BubbleField>

              <BubbleField label="GC Interval (ms)">
                <BubbleInput
                  type="number"
                  name="gc_interval_ms"
                  min="1000"
                  max="3600000"
                  value={(config.performanceConfig?.memory_management as any)?.gc_interval_ms || 60000}
                  onChange={handleMemoryManagementChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <div className="flex flex-wrap gap-2">
        <BubbleButton onClick={onValidate} variant="secondary">
          Validate performance config
        </BubbleButton>
        <BubbleButton onClick={() => onConfigUpdate({ performanceConfig: { ...config.performanceConfig } })}>
          Save performance config
        </BubbleButton>
      </div>
    </div>
  );
};

export const PerformanceConfigTab = withComponentBoundary(
  PerformanceConfigTabBase,
  'PerformanceConfigTab'
);
