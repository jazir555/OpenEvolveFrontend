// RAGBits Search Panel Component

import React, { useState } from 'react';
import { Search, Filter } from 'lucide-react';
import type { RAGBitsSearchPanelProps } from '../types/plugin-types';
import { useRAGBitsPlugin } from '../utils/createRAGBitsPlugin';
import { ragbitsLogger } from '../lib/structuredLogger';

export const RAGBitsSearchPanel: React.FC<RAGBitsSearchPanelProps> = ({
  initialQuery = '',
  onResult,
  onClose,
  showDebug = false
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [topK, setTopK] = useState(10);
  const [scoreThreshold, setScoreThreshold] = useState(0.7);
  const [enableHybridSearch, setEnableHybridSearch] = useState(true);
  const [enableReranking, setEnableReranking] = useState(true);
  const [isSearching, setIsSearching] = useState(false);

  const plugin = useRAGBitsPlugin();

  const handleSearch = async () => {
    if (!query.trim()) {
      return;
    }

    setIsSearching(true);

    const correlationId = `search-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

    try {
      ragbitsLogger.info('Starting RAGBits search', {
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

      ragbitsLogger.info('RAGBits search completed successfully', {
        correlation_id: correlationId,
        source_service: 'ragbits-plugin',
        results_count: result.results.length,
        execution_time: result.executionTime
      });

      onResult(result);
    } catch (error) {
      ragbitsLogger.error('RAGBits search failed', error as Error, {
        correlation_id: correlationId,
        source_service: 'ragbits-plugin',
        query_length: query.length
      });
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  return (
    <div className="ragbits-search-panel">
      <div className="search-header">
        <Search className="icon" />
        <h2>Knowledge Search</h2>
      </div>

      <div className="search-content">
        {/* Search Input */}
        <div className="search-input-section">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter your search query..."
            rows={3}
            className="search-textarea"
          />
          <button
            className="btn btn-primary search-button"
            onClick={handleSearch}
            disabled={isSearching || !query.trim()}
          >
            <Search className="icon" />
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Options */}
        <div className="search-options">
          <div className="options-header">
            <Filter className="icon" />
            <h3>Search Options</h3>
          </div>

          <div className="options-grid">
            <div className="form-group">
              <label>Top K Results</label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                min="1"
                max="100"
              />
            </div>

            <div className="form-group">
              <label>Score Threshold</label>
              <input
                type="number"
                value={scoreThreshold}
                onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
                min="0"
                max="1"
                step="0.1"
              />
            </div>

            <div className="form-group checkbox">
              <input
                type="checkbox"
                id="enableHybrid"
                checked={enableHybridSearch}
                onChange={(e) => setEnableHybridSearch(e.target.checked)}
              />
              <label htmlFor="enableHybrid">Hybrid Search</label>
            </div>

            <div className="form-group checkbox">
              <input
                type="checkbox"
                id="enableRerank"
                checked={enableReranking}
                onChange={(e) => setEnableReranking(e.target.checked)}
              />
              <label htmlFor="enableRerank">Enable Reranking</label>
            </div>
          </div>
        </div>

        {showDebug && (
          <div className="debug-info">
            <h4>Debug Information</h4>
            <pre>{JSON.stringify({
              query,
              topK,
              scoreThreshold,
              enableHybridSearch,
              enableReranking
            }, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="search-actions">
        <button className="btn btn-secondary" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};
