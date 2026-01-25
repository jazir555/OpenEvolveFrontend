// RAGBits Search Panel Component

import React, { useState } from 'react';
import { Search, Filter } from 'lucide-react';
import type { RAGBitsSearchPanelProps } from '../types/plugin-types';

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

  const handleSearch = async () => {
    if (!query.trim()) {
      return;
    }

    setIsSearching(true);

    try {
      // This would typically call the plugin's search method
      const result = {
        success: true,
        query,
        results: [],
        totalResults: 0,
        executionTime: 0,
        metadata: {
          searchType: enableHybridSearch ? 'hybrid' : 'semantic',
          vectorStoreUsed: 'unknown',
          rerankingApplied: enableReranking,
          cacheHit: false
        },
        errors: [],
        warnings: [],
        timestamp: new Date()
      };

      onResult(result);
    } catch (error) {
      console.error('Search failed:', error);
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
