// RAGBits Configuration Panel Component

import React, { useState, useEffect } from 'react';
import { Settings, Save, RotateCcw } from 'lucide-react';
import type { RAGBitsConfigPanelProps, RAGBitsPluginConfig } from '../types/plugin-types';
import { DEFAULT_RAGBITS_CONFIG } from '../types/plugin-types';

export const RAGBitsConfigPanel: React.FC<RAGBitsConfigPanelProps> = ({
  initialConfig,
  onSave,
  onCancel,
  showAdvanced = false
}) => {
  const [config, setConfig] = useState<RAGBitsPluginConfig>({
    ...DEFAULT_RAGBITS_CONFIG,
    ...initialConfig
  });

  const [showAdvancedOptions, setShowAdvancedOptions] = useState(showAdvanced);

  const handleSave = () => {
    onSave(config);
  };

  const handleReset = () => {
    setConfig(DEFAULT_RAGBITS_CONFIG);
  };

  return (
    <div className="ragbits-config-panel">
      <div className="config-header">
        <Settings className="icon" />
        <h2>RAGBits Configuration</h2>
      </div>

      <div className="config-content">
        {/* Server Configuration */}
        <div className="config-section">
          <h3>Server Settings</h3>
          <div className="form-group">
            <label>Server URL</label>
            <input
              type="text"
              value={config.serverUrl}
              onChange={(e) => setConfig({ ...config, serverUrl: e.target.value })}
              placeholder="http://localhost:3000/ragbits"
            />
          </div>
          <div className="form-group">
            <label>API Key (Optional)</label>
            <input
              type="password"
              value={config.apiKey || ''}
              onChange={(e) => setConfig({ ...config, apiKey: e.target.value })}
              placeholder="Enter API key if required"
            />
          </div>
          <div className="form-group">
            <label>Timeout (seconds)</label>
            <input
              type="number"
              value={config.timeout || 30}
              onChange={(e) => setConfig({ ...config, timeout: parseInt(e.target.value) })}
              min="1"
              max="300"
            />
          </div>
        </div>

        {/* Search Settings */}
        <div className="config-section">
          <h3>Search Settings</h3>
          <div className="form-group">
            <label>Default Top K Results</label>
            <input
              type="number"
              value={config.defaultTopK}
              onChange={(e) => setConfig({ ...config, defaultTopK: parseInt(e.target.value) })}
              min="1"
              max="100"
            />
          </div>
          <div className="form-group">
            <label>Score Threshold</label>
            <input
              type="number"
              value={config.defaultScoreThreshold}
              onChange={(e) => setConfig({ ...config, defaultScoreThreshold: parseFloat(e.target.value) })}
              min="0"
              max="1"
              step="0.1"
            />
          </div>
          <div className="form-group checkbox">
            <input
              type="checkbox"
              id="enableHybridSearch"
              checked={config.enableHybridSearch}
              onChange={(e) => setConfig({ ...config, enableHybridSearch: e.target.checked })}
            />
            <label htmlFor="enableHybridSearch">Enable Hybrid Search</label>
          </div>
          <div className="form-group checkbox">
            <input
              type="checkbox"
              id="enableReranking"
              checked={config.enableReranking}
              onChange={(e) => setConfig({ ...config, enableReranking: e.target.checked })}
            />
            <label htmlFor="enableReranking">Enable Reranking</label>
          </div>
        </div>

        {/* Indexing Settings */}
        <div className="config-section">
          <h3>Indexing Settings</h3>
          <div className="form-group checkbox">
            <input
              type="checkbox"
              id="autoIndexArtifacts"
              checked={config.autoIndexArtifacts}
              onChange={(e) => setConfig({ ...config, autoIndexArtifacts: e.target.checked })}
            />
            <label htmlFor="autoIndexArtifacts">Auto-Index Artifacts</label>
          </div>
          <div className="form-group">
            <label>Indexing Batch Size</label>
            <input
              type="number"
              value={config.indexingBatchSize}
              onChange={(e) => setConfig({ ...config, indexingBatchSize: parseInt(e.target.value) })}
              min="1"
              max="1000"
            />
          </div>
        </div>

        {/* Advanced Options */}
        <div className="config-section">
          <div className="section-header">
            <h3>Advanced Options</h3>
            <button
              type="button"
              className="toggle-button"
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            >
              {showAdvancedOptions ? 'Hide' : 'Show'}
            </button>
          </div>

          {showAdvancedOptions && (
            <>
              <div className="form-group checkbox">
                <input
                  type="checkbox"
                  id="enableCaching"
                  checked={config.enableCaching}
                  onChange={(e) => setConfig({ ...config, enableCaching: e.target.checked })}
                />
                <label htmlFor="enableCaching">Enable Caching</label>
              </div>
              <div className="form-group">
                <label>Cache TTL (seconds)</label>
                <input
                  type="number"
                  value={config.cacheTTLSeconds}
                  onChange={(e) => setConfig({ ...config, cacheTTLSeconds: parseInt(e.target.value) })}
                  min="60"
                  max="86400"
                />
              </div>
              <div className="form-group">
                <label>Max Search Time (seconds)</label>
                <input
                  type="number"
                  value={config.maxSearchTime}
                  onChange={(e) => setConfig({ ...config, maxSearchTime: parseInt(e.target.value) })}
                  min="1"
                  max="300"
                />
              </div>
            </>
          )}
        </div>
      </div>

      <div className="config-actions">
        <button type="button" className="btn btn-secondary" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleReset}
        >
          <RotateCcw className="icon" />
          Reset
        </button>
        <button type="button" className="btn btn-primary" onClick={handleSave}>
          <Save className="icon" />
          Save Configuration
        </button>
      </div>
    </div>
  );
};
