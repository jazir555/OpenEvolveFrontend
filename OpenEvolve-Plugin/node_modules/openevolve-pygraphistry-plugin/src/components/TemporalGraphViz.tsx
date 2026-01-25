import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { TemporalGraphResult } from '../types/plugin-types';

interface Props {
  initialQuery?: string;
}

export const TemporalGraphViz: React.FC<Props> = ({ 
  initialQuery = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.temporalGraphEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<TemporalGraphResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState(initialQuery);

  const searchTemporalGraph = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/graphiti/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Temporal search failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-purple-100 rounded-lg bg-purple-50/30 text-purple-400">
        <p className="font-medium italic">Temporal Knowledge Graph is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex gap-2">
        <input 
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && searchTemporalGraph()}
          placeholder="Search temporal facts..."
          className="flex-1 p-2 border rounded-md focus:ring-2 focus:ring-purple-500 outline-none text-sm"
        />
        <button
          onClick={searchTemporalGraph}
          disabled={loading || !query}
          className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 transition-colors text-sm font-medium"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="flex justify-between items-center px-1">
            <span className="text-xs font-bold text-slate-400 uppercase">Discovered Facts</span>
            <span className="text-[10px] text-slate-400">{result.edges.length} results</span>
          </div>
          
          <div className="grid grid-cols-1 gap-3">
            {result.edges.map((edge, i) => (
              <div key={i} className="p-3 border rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors border-l-4 border-l-purple-400">
                <div className="flex justify-between items-start mb-1">
                  <p className="text-sm font-medium text-slate-800">{edge.fact}</p>
                  {edge.valid_at && (
                    <span className="text-[10px] bg-white px-1.5 py-0.5 rounded border text-slate-500 font-mono">
                      {new Date(edge.valid_at).toLocaleDateString()}
                    </span>
                  )}
                </div>
                <div className="flex gap-2 items-center mt-2">
                  <span className="text-[10px] px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded-full font-bold">
                    {result.nodes.find(n => n.uuid === edge.source_node)?.name || 'Unknown'}
                  </span>
                  <span className="text-slate-300 text-[10px]">→</span>
                  <span className="text-[10px] px-1.5 py-0.5 bg-indigo-100 text-indigo-700 rounded-full font-bold">
                    {result.nodes.find(n => n.uuid === edge.target_node)?.name || 'Unknown'}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {result.edges.length === 0 && (
            <div className="py-12 text-center text-slate-400 border-2 border-dashed rounded-lg">
              No facts found for "{query}".
            </div>
          )}
        </div>
      )}
    </div>
  );
};
