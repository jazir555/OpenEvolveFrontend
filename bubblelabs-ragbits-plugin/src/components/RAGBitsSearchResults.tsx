// RAGBits Search Results Component

import React from 'react';
import { FileText, ExternalLink } from 'lucide-react';
import type { RAGBitsSearchResultsProps, RAGBitsSearchResult } from '../types/plugin-types';

export const RAGBitsSearchResults: React.FC<RAGBitsSearchResultsProps> = ({
  results,
  onResultClick,
  showMetadata = true,
  showScores = true
}) => {
  if (results.length === 0) {
    return (
      <div className="ragbits-search-results empty">
        <p>No results found</p>
      </div>
    );
  }

  return (
    <div className="ragbits-search-results">
      <div className="results-header">
        <h3>Search Results</h3>
        <span className="results-count">{results.length} results</span>
      </div>

      <div className="results-list">
        {results.map((result, index) => (
          <SearchResultItem
            key={result.documentId || index}
            result={result}
            onClick={onResultClick}
            showMetadata={showMetadata}
            showScores={showScores}
          />
        ))}
      </div>
    </div>
  );
};

interface SearchResultItemProps {
  result: RAGBitsSearchResult;
  onClick?: (result: RAGBitsSearchResult) => void;
  showMetadata: boolean;
  showScores: boolean;
}

const SearchResultItem: React.FC<SearchResultItemProps> = ({
  result,
  onClick,
  showMetadata,
  showScores
}) => {
  const handleClick = () => {
    if (onClick) {
      onClick(result);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'green';
    if (score >= 0.6) return 'yellow';
    return 'red';
  };

  return (
    <div
      className={`search-result-item ${onClick ? 'clickable' : ''}`}
      onClick={handleClick}
    >
      <div className="result-header">
        <FileText className="result-icon" />
        <span className="result-type">{result.metadata.documentType}</span>
        {showScores && (
          <span className={`result-score score-${getScoreColor(result.relevanceScore)}`}>
            {(result.relevanceScore * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="result-content">
        {result.content.substring(0, 200)}
        {result.content.length > 200 && '...'}
      </div>

      {showMetadata && (
        <div className="result-metadata">
          {result.metadata.source && (
            <span className="metadata-item">Source: {result.metadata.source}</span>
          )}
          {result.metadata.stage && (
            <span className="metadata-item">Stage: {result.metadata.stage}</span>
          )}
          {result.metadata.team && (
            <span className="metadata-item">Team: {result.metadata.team}</span>
          )}
          {result.metadata.tags && result.metadata.tags.length > 0 && (
            <div className="metadata-tags">
              {result.metadata.tags.map((tag, i) => (
                <span key={i} className="tag">{tag}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {onClick && (
        <div className="result-action">
          <ExternalLink className="icon" />
        </div>
      )}
    </div>
  );
};
