import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { MinedPatternCluster } from '../types/plugin-types';

export const PatternMinerViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.patternMiningEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [clusters, setClusters] = useState<MinedPatternCluster[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchClusters = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/patterns/mined');
      if (!response.ok) {
        throw new Error('Failed to fetch mined patterns');
      }
      const data = await response.json();
      setClusters(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchClusters();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-amber-100 rounded-lg bg-amber-50/30 text-amber-400">
        <p className="font-medium italic">Pattern Mining visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-amber-500 flex items-center justify-center text-white font-bold text-xs shadow-sm">M</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Solution Pattern Discovery</h3>
        </div>
        <button 
          onClick={fetchClusters}
          disabled={loading}
          className="text-xs font-bold text-amber-600 hover:underline px-2 py-1"
        >
          {loading ? 'Mining...' : 'Sync Patterns'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 animate-in fade-in slide-in-from-top-2">
        {clusters.map((c) => (
          <div key={c.cluster_id} className="p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all flex flex-col gap-3">
            <div className="flex justify-between items-center">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Cluster #{c.cluster_id}</span>
              <span className="text-[10px] font-bold bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full border border-amber-200">
                {c.size} Patterns
              </span>
            </div>
            
            <div>
              <h4 className="text-sm font-bold text-slate-800 capitalize">{c.most_common_domain.replace('_', ' ')}</h4>
              <p className="text-xs text-slate-500 mt-1 line-clamp-2 italic">"{c.description}"</p>
            </div>

            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-200/50">
              <div>
                <p className="text-[8px] font-bold text-slate-400 uppercase">Avg Complexity</p>
                <p className="text-sm font-mono font-bold text-slate-700">{c.avg_complexity.toFixed(1)}/10</p>
              </div>
              <div className="text-right">
                <p className="text-[8px] font-bold text-slate-400 uppercase">Avg Success</p>
                <p className="text-sm font-mono font-bold text-emerald-600">{(c.avg_success_rate * 100).toFixed(0)}%</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
