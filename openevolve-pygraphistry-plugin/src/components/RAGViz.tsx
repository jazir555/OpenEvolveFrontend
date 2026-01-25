import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { RAGSearchResult } from '../types/plugin-types';

interface Props {
  initialQuery?: string;
}

export const RAGViz: React.FC<Props> = ({ 
  initialQuery = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.ragEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<RAGSearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState(initialQuery);

  const runSearch = async () => {
    if (!isEnabled || !query) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/rag/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        throw new Error('RAG search failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-sky-100 rounded-lg bg-sky-50/30 text-sky-400">
        <p className="font-medium italic">Knowledge Retrieval (RAG) visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-sky-700 flex items-center justify-center text-white font-bold text-xs shadow-sm">R</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Contextual Knowledge Retrieval (RAG)</h3>
        </div>
        <div className="flex gap-2">
          <input 
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runSearch()}
            placeholder="Query knowledge base (e.g., 'What are the RESE phase 4 requirements?')..."
            className="flex-1 p-2 border rounded-md focus:ring-2 focus:ring-sky-500 outline-none text-sm"
          />
          <button
            onClick={runSearch}
            disabled={loading || !query}
            className="px-4 py-2 bg-sky-700 text-white rounded hover:bg-sky-800 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Retrieved Context Segments</h4>
          {results.map((res, i) => (
            <div key={i} className="p-3 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all group">
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded-full border border-sky-100">
                  Relevance: {(res.score * 100).toFixed(1)}%
                </span>
                <span className="text-[9px] text-slate-400 font-mono italic">Source: {res.source}</span>
              </div>
              <p className="text-xs text-slate-700 leading-relaxed font-sans">{res.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
