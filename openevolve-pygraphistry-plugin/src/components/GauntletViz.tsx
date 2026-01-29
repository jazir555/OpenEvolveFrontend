import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GauntletEffectivenessResult } from '../types/plugin-types';

export const GauntletViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.gauntletEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<GauntletEffectivenessResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchEffectiveness = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/gauntlet/effectiveness', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gauntlet_ids: ["Red-Team-Core", "Gold-Team-Verify"] })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch gauntlet data');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchEffectiveness();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-rose-100 rounded-lg bg-rose-50/30 text-rose-400">
        <p className="font-medium italic">Gauntlet Effectiveness visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-rose-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">G</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Gauntlet Effectiveness</h3>
        </div>
        <button 
          onClick={fetchEffectiveness}
          disabled={loading}
          className="text-xs font-bold text-rose-600 hover:underline"
        >
          {loading ? 'Analyzing...' : 'Refresh Stats'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2">
        {result && Object.values(result).map((g) => (
          <div key={g.gauntlet_id} className="p-4 border rounded-xl bg-slate-50 relative overflow-hidden group">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h4 className="text-sm font-bold text-slate-800">{g.gauntlet_id}</h4>
                <p className="text-[10px] text-slate-400 uppercase font-bold tracking-widest">{g.gauntlet_type}</p>
              </div>
              <div className="text-right">
                <p className="text-[8px] font-bold text-slate-400 uppercase">Effectiveness</p>
                <p className="text-xl font-mono font-bold text-rose-600">{(g.effectiveness_score * 100).toFixed(1)}%</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="space-y-1">
                <p className="text-[10px] font-bold text-slate-500 uppercase">Catch Rate</p>
                <div className="w-full bg-slate-200 h-1 rounded-full overflow-hidden">
                  <div className="bg-emerald-500 h-full" style={{ width: `${g.avg_catch_rate * 100}%` }} />
                </div>
                <p className="text-[10px] font-mono font-bold text-slate-700">{(g.avg_catch_rate * 100).toFixed(1)}%</p>
              </div>
              <div className="space-y-1">
                <p className="text-[10px] font-bold text-slate-500 uppercase">False Positives</p>
                <div className="w-full bg-slate-200 h-1 rounded-full overflow-hidden">
                  <div className="bg-rose-400 h-full" style={{ width: `${g.avg_false_positive_rate * 100}%` }} />
                </div>
                <p className="text-[10px] font-mono font-bold text-slate-700">{(g.avg_false_positive_rate * 100).toFixed(1)}%</p>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-200/50 flex justify-between items-center">
              <span className="text-[10px] text-slate-400 font-medium">{g.total_runs} Total Executions</span>
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[8px] font-bold text-emerald-600 uppercase">Optimal</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
