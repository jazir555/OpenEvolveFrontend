import { useState } from 'react';
import { Database, Search, Loader2, AlertCircle } from 'lucide-react';
import { toast } from 'react-toastify';
import {
  searchKnowledge,
  isRagbitsAvailable,
} from '../services/ragbitsService';
import { RagbitsSearchResult } from '../lib/ragbitsClient';

export interface RagbitsKnowledgeSearchProps {
  initialQuery?: string;
  filters?: Record<string, unknown>;
  topK?: number;
  onResults?: (results: RagbitsSearchResult[]) => void;
  className?: string;
}

export function RagbitsKnowledgeSearch({
  initialQuery = '',
  filters,
  topK = 5,
  onResults,
  className = '',
}: RagbitsKnowledgeSearchProps) {
  const [query, setQuery] = useState(initialQuery);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<RagbitsSearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!isRagbitsAvailable()) {
      toast.error('RAGBits service is not available. Please configure RAGBits integration.');
      return;
    }

    if (!query.trim()) {
      toast.warning('Please enter a query to search the knowledge base.');
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const response = await searchKnowledge({
        query,
        filters,
        topK,
      });

      if (!response.success) {
        const message = response.error || 'RAGBits search failed';
        setError(message);
        toast.error(message);
        return;
      }

      setResults(response.results || []);
      if (onResults) {
        onResults(response.results || []);
      }

      toast.success(`RAGBits returned ${response.results.length} result(s)`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      toast.error(`RAGBits error: ${message}`);
      console.error('RAGBits search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className={`border rounded-lg p-4 bg-background/50 backdrop-blur-sm ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5 text-emerald-500" />
          <h3 className="font-semibold text-sm">RAGBits Knowledge Search</h3>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          <span>{isSearching ? 'Searching...' : 'Ready'}</span>
        </div>
      </div>

      <div className="mb-3">
        <label className="text-xs text-muted-foreground mb-2 block">Query</label>
        <input
          value={query}
          onChange={(event: any) => setQuery(event.target.value)}
          placeholder="Search for patterns, solutions, or artifacts..."
          className="w-full rounded-md border border-muted bg-muted/20 px-3 py-2 text-sm"
        />
      </div>

      <button
        onClick={handleSearch}
        disabled={isSearching}
        className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
          isSearching
            ? 'bg-muted text-muted-foreground cursor-not-allowed'
            : 'bg-emerald-600 text-white hover:bg-emerald-700'
        }`}
      >
        {isSearching ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Searching...
          </>
        ) : (
          <>
            <Search className="h-4 w-4" />
            Search Knowledge
          </>
        )}
      </button>

      {error && (
        <div className="mt-3 bg-destructive/10 border border-destructive/20 rounded-md p-2 text-xs text-destructive flex gap-2">
          <AlertCircle className="h-4 w-4 mt-0.5" />
          <div>
            <p className="font-medium">Error</p>
            <p>{error}</p>
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="mt-3 border-t pt-3">
          <h4 className="font-medium text-sm mb-2">Results</h4>
          <div className="space-y-2 max-h-64 overflow-auto">
            {results.map((result: RagbitsSearchResult, index: number) => (
              <div key={`${index}-${result.content.slice(0, 12)}`} className="bg-muted/20 rounded-md p-2 text-xs">
                <div className="text-muted-foreground mb-1">
                  Score: {result.score?.toFixed(3) ?? 'n/a'}
                </div>
                <div className="whitespace-pre-wrap break-words">{result.content}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
