import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from 'react';
import { Database, Search, Loader2, AlertCircle } from 'lucide-react';
import { toast } from 'react-toastify';
import { searchKnowledge, isRagbitsAvailable, } from '../services/ragbitsService';
export function RagbitsKnowledgeSearch({ initialQuery = '', filters, topK = 5, onResults, className = '', }) {
    const [query, setQuery] = useState(initialQuery);
    const [isSearching, setIsSearching] = useState(false);
    const [results, setResults] = useState([]);
    const [error, setError] = useState(null);
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
        }
        catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
            toast.error(`RAGBits error: ${message}`);
            console.error('RAGBits search error:', err);
        }
        finally {
            setIsSearching(false);
        }
    };
    return (_jsxs("div", { className: `border rounded-lg p-4 bg-background/50 backdrop-blur-sm ${className}`, children: [_jsxs("div", { className: "flex items-center justify-between mb-3", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx(Database, { className: "h-5 w-5 text-emerald-500" }), _jsx("h3", { className: "font-semibold text-sm", children: "RAGBits Knowledge Search" })] }), _jsxs("div", { className: "flex items-center gap-2 text-xs text-muted-foreground", children: [isSearching ? _jsx(Loader2, { className: "h-4 w-4 animate-spin" }) : _jsx(Search, { className: "h-4 w-4" }), _jsx("span", { children: isSearching ? 'Searching...' : 'Ready' })] })] }), _jsxs("div", { className: "mb-3", children: [_jsx("label", { className: "text-xs text-muted-foreground mb-2 block", children: "Query" }), _jsx("input", { value: query, onChange: (event) => setQuery(event.target.value), placeholder: "Search for patterns, solutions, or artifacts...", className: "w-full rounded-md border border-muted bg-muted/20 px-3 py-2 text-sm" })] }), _jsx("button", { onClick: handleSearch, disabled: isSearching, className: `w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${isSearching
                    ? 'bg-muted text-muted-foreground cursor-not-allowed'
                    : 'bg-emerald-600 text-white hover:bg-emerald-700'}`, children: isSearching ? (_jsxs(_Fragment, { children: [_jsx(Loader2, { className: "h-4 w-4 animate-spin" }), "Searching..."] })) : (_jsxs(_Fragment, { children: [_jsx(Search, { className: "h-4 w-4" }), "Search Knowledge"] })) }), error && (_jsxs("div", { className: "mt-3 bg-destructive/10 border border-destructive/20 rounded-md p-2 text-xs text-destructive flex gap-2", children: [_jsx(AlertCircle, { className: "h-4 w-4 mt-0.5" }), _jsxs("div", { children: [_jsx("p", { className: "font-medium", children: "Error" }), _jsx("p", { children: error })] })] })), results.length > 0 && (_jsxs("div", { className: "mt-3 border-t pt-3", children: [_jsx("h4", { className: "font-medium text-sm mb-2", children: "Results" }), _jsx("div", { className: "space-y-2 max-h-64 overflow-auto", children: results.map((result, index) => (_jsxs("div", { className: "bg-muted/20 rounded-md p-2 text-xs", children: [_jsxs("div", { className: "text-muted-foreground mb-1", children: ["Score: ", result.score?.toFixed(3) ?? 'n/a'] }), _jsx("div", { className: "whitespace-pre-wrap break-words", children: result.content })] }, `${index}-${result.content.slice(0, 12)}`))) })] }))] }));
}
//# sourceMappingURL=RagbitsKnowledgeSearch.js.map