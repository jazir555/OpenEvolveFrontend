"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RagbitsKnowledgeSearch = RagbitsKnowledgeSearch;
const react_1 = require("react");
const lucide_react_1 = require("lucide-react");
const react_toastify_1 = require("react-toastify");
const ragbitsService_1 = require("../services/ragbitsService");
function RagbitsKnowledgeSearch({ initialQuery = '', filters, topK = 5, onResults, className = '', }) {
    const [query, setQuery] = (0, react_1.useState)(initialQuery);
    const [isSearching, setIsSearching] = (0, react_1.useState)(false);
    const [results, setResults] = (0, react_1.useState)([]);
    const [error, setError] = (0, react_1.useState)(null);
    const handleSearch = async () => {
        if (!(0, ragbitsService_1.isRagbitsAvailable)()) {
            react_toastify_1.toast.error('RAGBits service is not available. Please configure RAGBits integration.');
            return;
        }
        if (!query.trim()) {
            react_toastify_1.toast.warning('Please enter a query to search the knowledge base.');
            return;
        }
        setIsSearching(true);
        setError(null);
        try {
            const response = await (0, ragbitsService_1.searchKnowledge)({
                query,
                filters,
                topK,
            });
            if (!response.success) {
                const message = response.error || 'RAGBits search failed';
                setError(message);
                react_toastify_1.toast.error(message);
                return;
            }
            setResults(response.results || []);
            if (onResults) {
                onResults(response.results || []);
            }
            react_toastify_1.toast.success(`RAGBits returned ${response.results.length} result(s)`);
        }
        catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
            react_toastify_1.toast.error(`RAGBits error: ${message}`);
            console.error('RAGBits search error:', err);
        }
        finally {
            setIsSearching(false);
        }
    };
    return (<div className={`border rounded-lg p-4 bg-background/50 backdrop-blur-sm ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <lucide_react_1.Database className="h-5 w-5 text-emerald-500"/>
          <h3 className="font-semibold text-sm">RAGBits Knowledge Search</h3>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {isSearching ? <lucide_react_1.Loader2 className="h-4 w-4 animate-spin"/> : <lucide_react_1.Search className="h-4 w-4"/>}
          <span>{isSearching ? 'Searching...' : 'Ready'}</span>
        </div>
      </div>

      <div className="mb-3">
        <label className="text-xs text-muted-foreground mb-2 block">Query</label>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search for patterns, solutions, or artifacts..." className="w-full rounded-md border border-muted bg-muted/20 px-3 py-2 text-sm"/>
      </div>

      <button onClick={handleSearch} disabled={isSearching} className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${isSearching
            ? 'bg-muted text-muted-foreground cursor-not-allowed'
            : 'bg-emerald-600 text-white hover:bg-emerald-700'}`}>
        {isSearching ? (<>
            <lucide_react_1.Loader2 className="h-4 w-4 animate-spin"/>
            Searching...
          </>) : (<>
            <lucide_react_1.Search className="h-4 w-4"/>
            Search Knowledge
          </>)}
      </button>

      {error && (<div className="mt-3 bg-destructive/10 border border-destructive/20 rounded-md p-2 text-xs text-destructive flex gap-2">
          <lucide_react_1.AlertCircle className="h-4 w-4 mt-0.5"/>
          <div>
            <p className="font-medium">Error</p>
            <p>{error}</p>
          </div>
        </div>)}

      {results.length > 0 && (<div className="mt-3 border-t pt-3">
          <h4 className="font-medium text-sm mb-2">Results</h4>
          <div className="space-y-2 max-h-64 overflow-auto">
            {results.map((result, index) => (<div key={`${index}-${result.content.slice(0, 12)}`} className="bg-muted/20 rounded-md p-2 text-xs">
                <div className="text-muted-foreground mb-1">
                  Score: {result.score?.toFixed(3) ?? 'n/a'}
                </div>
                <div className="whitespace-pre-wrap break-words">{result.content}</div>
              </div>))}
          </div>
        </div>)}
    </div>);
}
//# sourceMappingURL=RagbitsKnowledgeSearch.js.map