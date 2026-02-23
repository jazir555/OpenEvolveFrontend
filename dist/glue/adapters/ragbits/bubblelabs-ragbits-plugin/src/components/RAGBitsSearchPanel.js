"use strict";
// RAGBits Search Panel Component
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
exports.RAGBitsSearchPanel = void 0;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const createRAGBitsPlugin_1 = require("../utils/createRAGBitsPlugin");
const structuredLogger_1 = require("../lib/structuredLogger");
const RAGBitsSearchPanel = ({ initialQuery = '', onResult, onClose, showDebug = false }) => {
    const [query, setQuery] = (0, react_1.useState)(initialQuery);
    const [topK, setTopK] = (0, react_1.useState)(10);
    const [scoreThreshold, setScoreThreshold] = (0, react_1.useState)(0.7);
    const [enableHybridSearch, setEnableHybridSearch] = (0, react_1.useState)(true);
    const [enableReranking, setEnableReranking] = (0, react_1.useState)(true);
    const [isSearching, setIsSearching] = (0, react_1.useState)(false);
    const plugin = (0, createRAGBitsPlugin_1.useRAGBitsPlugin)();
    const handleSearch = async () => {
        if (!query.trim()) {
            return;
        }
        setIsSearching(true);
        const correlationId = `search-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        try {
            structuredLogger_1.ragbitsLogger.info('Starting RAGBits search', {
                correlation_id: correlationId,
                source_service: 'ragbits-plugin',
                target_service: 'ragbits-server',
                query_length: query.length,
                top_k: topK,
                score_threshold: scoreThreshold
            });
            // Call the plugin's search method
            const result = await plugin.search({
                query,
                topK,
                scoreThreshold,
                enableHybridSearch,
                enableReranking
            });
            structuredLogger_1.ragbitsLogger.info('RAGBits search completed successfully', {
                correlation_id: correlationId,
                source_service: 'ragbits-plugin',
                results_count: result.results.length,
                execution_time: result.executionTime
            });
            onResult(result);
        }
        catch (error) {
            structuredLogger_1.ragbitsLogger.error('RAGBits search failed', error, {
                correlation_id: correlationId,
                source_service: 'ragbits-plugin',
                query_length: query.length
            });
        }
        finally {
            setIsSearching(false);
        }
    };
    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSearch();
        }
    };
    return (<div className="ragbits-search-panel">
      <div className="search-header">
        <lucide_react_1.Search className="icon"/>
        <h2>Knowledge Search</h2>
      </div>

      <div className="search-content">
        {/* Search Input */}
        <div className="search-input-section">
          <textarea value={query} onChange={(e) => setQuery(e.target.value)} onKeyPress={handleKeyPress} placeholder="Enter your search query..." rows={3} className="search-textarea"/>
          <button className="btn btn-primary search-button" onClick={handleSearch} disabled={isSearching || !query.trim()}>
            <lucide_react_1.Search className="icon"/>
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Options */}
        <div className="search-options">
          <div className="options-header">
            <lucide_react_1.Filter className="icon"/>
            <h3>Search Options</h3>
          </div>

          <div className="options-grid">
            <div className="form-group">
              <label>Top K Results</label>
              <input type="number" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} min="1" max="100"/>
            </div>

            <div className="form-group">
              <label>Score Threshold</label>
              <input type="number" value={scoreThreshold} onChange={(e) => setScoreThreshold(parseFloat(e.target.value))} min="0" max="1" step="0.1"/>
            </div>

            <div className="form-group checkbox">
              <input type="checkbox" id="enableHybrid" checked={enableHybridSearch} onChange={(e) => setEnableHybridSearch(e.target.checked)}/>
              <label htmlFor="enableHybrid">Hybrid Search</label>
            </div>

            <div className="form-group checkbox">
              <input type="checkbox" id="enableRerank" checked={enableReranking} onChange={(e) => setEnableReranking(e.target.checked)}/>
              <label htmlFor="enableRerank">Enable Reranking</label>
            </div>
          </div>
        </div>

        {showDebug && (<div className="debug-info">
            <h4>Debug Information</h4>
            <pre>{JSON.stringify({
                query,
                topK,
                scoreThreshold,
                enableHybridSearch,
                enableReranking
            }, null, 2)}</pre>
          </div>)}
      </div>

      <div className="search-actions">
        <button className="btn btn-secondary" onClick={onClose}>
          Close
        </button>
      </div>
    </div>);
};
exports.RAGBitsSearchPanel = RAGBitsSearchPanel;
//# sourceMappingURL=RAGBitsSearchPanel.js.map