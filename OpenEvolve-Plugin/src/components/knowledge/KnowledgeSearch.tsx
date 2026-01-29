import { useState } from 'react';
import { cn } from '@/lib/utils';
import { BubbleBadge, BubbleButton, BubbleField, BubbleInput, BubbleSelect } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface SearchResult {
  id: string;
  title: string;
  snippet: string;
  type: string;
  score: number;
}

interface KnowledgeSearchProps {
  onSearch: (query: string, filters?: SearchFilters) => Promise<SearchResult[]>;
  onResultClick?: (result: SearchResult) => void;
  className?: string;
}

export interface SearchFilters {
  types?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
}

function KnowledgeSearchBase({ onSearch, onResultClick, className }: KnowledgeSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    try {
      const searchResults = await onSearch(query);
      setResults(Array.isArray(searchResults) ? searchResults : []);
    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Search failed' } });
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className={cn('knowledge-search', className)}>
      {/* Search Input */}
      <div className="flex gap-2 mb-4">
        <div className="flex-1 relative">
          <BubbleInput
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search knowledge base..."
            className="pl-10"
          />
          <svg
            className="absolute left-3 top-2.5 h-5 w-5 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
        <BubbleButton
          onClick={handleSearch}
          disabled={isSearching || !query.trim()}
        >
          {isSearching ? 'Searching...' : 'Search'}
        </BubbleButton>
        <BubbleButton
          onClick={() => setShowAdvanced(!showAdvanced)}
          variant="secondary"
        >
          Advanced
        </BubbleButton>
      </div>

      {/* Advanced Filters */}
      {showAdvanced && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Filters</h3>
          <div className="grid grid-cols-2 gap-4">
            <BubbleField label="Content Types">
              <BubbleSelect>
                <option>All Types</option>
                <option>Evolutions</option>
                <option>Artifacts</option>
                <option>Workflows</option>
              </BubbleSelect>
            </BubbleField>
            <BubbleField label="Date Range">
              <BubbleSelect>
                <option>All Time</option>
                <option>Last 7 Days</option>
                <option>Last 30 Days</option>
                <option>Last Year</option>
              </BubbleSelect>
            </BubbleField>
          </div>
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-gray-600">Found {results.length} results</p>
          {results.map((result) => (
            <div
              key={result.id}
              onClick={() => onResultClick?.(result)}
              className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="text-lg font-semibold text-gray-900">{result.title}</h4>
                <BubbleBadge tone="info">{result.type}</BubbleBadge>
              </div>
              <p className="text-sm text-gray-600 mb-2">{result.snippet}</p>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span>Relevance: {(Number(result.score ?? 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {query && results.length === 0 && !isSearching && (
        <div className="text-center py-12 text-gray-500">
          <p>No results found for "{query}"</p>
        </div>
      )}
    </div>
  );
}

export const KnowledgeSearch = withComponentBoundary(KnowledgeSearchBase, 'KnowledgeSearch');
