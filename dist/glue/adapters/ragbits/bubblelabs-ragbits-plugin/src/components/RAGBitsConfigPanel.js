"use strict";
// RAGBits Configuration Panel Component
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAGBitsConfigPanel = void 0;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const plugin_types_1 = require("../types/plugin-types");
const RAGBitsConfigPanel = ({ initialConfig, onSave, onCancel, showAdvanced = false }) => {
    const [config, setConfig] = (0, react_1.useState)({
        ...plugin_types_1.DEFAULT_RAGBITS_CONFIG,
        ...initialConfig
    });
    const [showAdvancedOptions, setShowAdvancedOptions] = (0, react_1.useState)(showAdvanced);
    const handleSave = () => {
        onSave(config);
    };
    const handleReset = () => {
        setConfig(plugin_types_1.DEFAULT_RAGBITS_CONFIG);
    };
    return (<div className="ragbits-config-panel">
      <div className="config-header">
        <lucide_react_1.Settings className="icon"/>
        <h2>RAGBits Configuration</h2>
      </div>

      <div className="config-content">
        {/* Server Configuration */}
        <div className="config-section">
          <h3>Server Settings</h3>
          <div className="form-group">
            <label>Server URL</label>
            <input type="text" value={config.serverUrl} onChange={(e) => setConfig({ ...config, serverUrl: e.target.value })} placeholder="http://localhost:3000/ragbits"/>
          </div>
          <div className="form-group">
            <label>API Key (Optional)</label>
            <input type="password" value={config.apiKey || ''} onChange={(e) => setConfig({ ...config, apiKey: e.target.value })} placeholder="Enter API key if required"/>
          </div>
          <div className="form-group">
            <label>Timeout (seconds)</label>
            <input type="number" value={config.timeout || 30} onChange={(e) => {
            const value = parseInt(e.target.value);
            setConfig({ ...config, timeout: isNaN(value) ? 30 : value });
        }} min="1" max="300"/>
          </div>
        </div>

        {/* Search Settings */}
        <div className="config-section">
          <h3>Search Settings</h3>
          <div className="form-group">
            <label>Default Top K Results</label>
            <input type="number" value={config.defaultTopK} onChange={(e) => {
            const value = parseInt(e.target.value);
            setConfig({ ...config, defaultTopK: isNaN(value) ? 10 : value });
        }} min="1" max="100"/>
          </div>
          <div className="form-group">
            <label>Score Threshold</label>
            <input type="number" value={config.defaultScoreThreshold} onChange={(e) => {
            const value = parseFloat(e.target.value);
            setConfig({ ...config, defaultScoreThreshold: isNaN(value) ? 0.7 : value });
        }} min="0" max="1" step="0.1"/>
          </div>
          <div className="form-group checkbox">
            <input type="checkbox" id="enableHybridSearch" checked={config.enableHybridSearch} onChange={(e) => setConfig({ ...config, enableHybridSearch: e.target.checked })}/>
            <label htmlFor="enableHybridSearch">Enable Hybrid Search</label>
          </div>
          <div className="form-group checkbox">
            <input type="checkbox" id="enableReranking" checked={config.enableReranking} onChange={(e) => setConfig({ ...config, enableReranking: e.target.checked })}/>
            <label htmlFor="enableReranking">Enable Reranking</label>
          </div>
        </div>

        {/* Indexing Settings */}
        <div className="config-section">
          <h3>Indexing Settings</h3>
          <div className="form-group checkbox">
            <input type="checkbox" id="autoIndexArtifacts" checked={config.autoIndexArtifacts} onChange={(e) => setConfig({ ...config, autoIndexArtifacts: e.target.checked })}/>
            <label htmlFor="autoIndexArtifacts">Auto-Index Artifacts</label>
          </div>
          <div className="form-group">
            <label>Indexing Batch Size</label>
            <input type="number" value={config.indexingBatchSize} onChange={(e) => {
            const value = parseInt(e.target.value);
            setConfig({ ...config, indexingBatchSize: isNaN(value) ? 100 : value });
        }} min="1" max="1000"/>
          </div>
        </div>

        {/* Advanced Options */}
        <div className="config-section">
          <div className="section-header">
            <h3>Advanced Options</h3>
            <button type="button" className="toggle-button" onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}>
              {showAdvancedOptions ? 'Hide' : 'Show'}
            </button>
          </div>

          {showAdvancedOptions && (<>
              <div className="form-group checkbox">
                <input type="checkbox" id="enableCaching" checked={config.enableCaching} onChange={(e) => setConfig({ ...config, enableCaching: e.target.checked })}/>
                <label htmlFor="enableCaching">Enable Caching</label>
              </div>
              <div className="form-group">
                <label>Cache TTL (seconds)</label>
                <input type="number" value={config.cacheTTLSeconds} onChange={(e) => {
                const value = parseInt(e.target.value);
                setConfig({ ...config, cacheTTLSeconds: isNaN(value) ? 3600 : value });
            }} min="60" max="86400"/>
              </div>
              <div className="form-group">
                <label>Max Search Time (seconds)</label>
                <input type="number" value={config.maxSearchTime} onChange={(e) => {
                const value = parseInt(e.target.value);
                setConfig({ ...config, maxSearchTime: isNaN(value) ? 15 : value });
            }} min="1" max="300"/>
              </div>
            </>)}
        </div>
      </div>

      <div className="config-actions">
        <button type="button" className="btn btn-secondary" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleReset}>
          <lucide_react_1.RotateCcw className="icon"/>
          Reset
        </button>
        <button type="button" className="btn btn-primary" onClick={handleSave}>
          <lucide_react_1.Save className="icon"/>
          Save Configuration
        </button>
      </div>
    </div>);
};
exports.RAGBitsConfigPanel = RAGBitsConfigPanel;
//# sourceMappingURL=RAGBitsConfigPanel.js.map