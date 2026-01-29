import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { MAPElitesResult } from '../types/plugin-types';

interface Props {
  initialContent?: string;
  iterations?: number;
}

export const MAPElitesViz: React.FC<Props> = ({ 
  initialContent = '',
  iterations = 50
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.mapElitesEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<MAPElitesResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [content, setContent] = useState(initialContent);

  const runEvolution = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/evolution/map-elites', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, iterations })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Evolution failed');
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
      <div className="p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400">
        <p className="font-medium italic">Quality-Diversity visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gradient-to-tr from-indigo-600 to-violet-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">QD</div>
          <h3 className="text-lg font-bold text-slate-800">Quality-Diversity Optimization (MAP-Elites)</h3>
        </div>
        <textarea 
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Enter code or content to optimize..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-indigo-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runEvolution}
            disabled={loading || !content}
            className="px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
          >
            {loading ? 'Evolving...' : 'Run QD Evolution'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Feature Space Grid Visualization (Simplified representation) */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">
                MAP-Elites Archive ({result.feature_dimensions.join(' vs ')})
              </h4>
              <div className="aspect-square border rounded-lg bg-slate-900 p-2 grid grid-cols-10 grid-rows-10 gap-0.5">
                {result.map_elites_grid.flat().map((val, i) => (
                  <div 
                    key={i} 
                    className="w-full h-full rounded-sm transition-colors hover:ring-1 hover:ring-white cursor-help"
                    style={{ 
                      backgroundColor: val > 0 ? `rgba(99, 102, 241, ${val})` : 'rgba(30, 41, 59, 0.5)',
                      opacity: val > 0 ? 1 : 0.2
                    }}
                    title={`Performance: ${(val * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <p className="text-[10px] text-center text-slate-400 italic">Heatmap represents high-performing individuals across the feature space.</p>
            </div>

            {/* Metrics History */}
            <div className="space-y-4">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Performance over Generations</h4>
              <div className="h-48 border rounded-lg bg-slate-50 relative flex items-end p-2 gap-1 overflow-hidden">
                {result.best_scores.map((score, i) => (
                  <div 
                    key={i} 
                    className="flex-1 bg-indigo-500 rounded-t-sm min-w-[2px]"
                    style={{ height: `${score * 100}%` }}
                  />
                ))}
                <div className="absolute inset-0 flex flex-col justify-between p-2 pointer-events-none">
                  <span className="text-[8px] font-bold text-slate-400 border-b border-dashed w-full">MAX</span>
                  <span className="text-[8px] font-bold text-slate-400 border-b border-dashed w-full">AVG</span>
                  <span className="text-[8px] font-bold text-slate-400 w-full">START</span>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 border rounded bg-white text-center">
                  <p className="text-[8px] font-bold text-slate-400 uppercase">Archive</p>
                  <p className="text-sm font-mono font-bold text-indigo-600">{result.map_elites_grid.flat().filter(v => v > 0).length}</p>
                </div>
                <div className="p-2 border rounded bg-white text-center">
                  <p className="text-[8px] font-bold text-slate-400 uppercase">Best</p>
                  <p className="text-sm font-mono font-bold text-emerald-600">{Math.max(...result.best_scores).toFixed(3)}</p>
                </div>
                <div className="p-2 border rounded bg-white text-center">
                  <p className="text-[8px] font-bold text-slate-400 uppercase">Diversity</p>
                  <p className="text-sm font-mono font-bold text-violet-600">{result.diversity_scores[result.diversity_scores.length-1].toFixed(3)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
