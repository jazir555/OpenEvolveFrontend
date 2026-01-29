import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { AdaptationResult } from '../types/plugin-types';

export const AdaptationViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.adaptationEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AdaptationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAdaptationStats = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/gauntlet/adaptation');
      if (!response.ok) {
        throw new Error('Failed to fetch adaptation statistics');
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
      fetchAdaptationStats();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-rose-100 rounded-lg bg-rose-50/30 text-rose-400">
        <p className="font-medium italic">Dynamic Adaptation visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-rose-500 to-amber-500 flex items-center justify-center text-white font-bold text-xs shadow-md">D</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Dynamic Gauntlet Adaptation</h3>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
          <span className="text-[10px] font-bold text-rose-600 uppercase">Optimization Engine</span>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-900 text-white rounded-xl shadow-lg border border-slate-800 flex flex-col items-center justify-center">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Total Adaptations</p>
              <p className="text-3xl font-mono font-bold text-rose-400 mt-1">{result.total_adaptations}</p>
            </div>
            
            <div className="md:col-span-2 p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 text-center">Strictness Optimization Distribution</h4>
              <div className="flex items-center h-8 w-full rounded-full overflow-hidden bg-slate-200 shadow-inner">
                <div 
                  className="bg-rose-500 h-full flex items-center justify-center text-[8px] font-bold text-white transition-all"
                  style={{ width: `${(result.strictness_distribution.more_strict / (result.total_adaptations || 1)) * 100}%` }}
                  title="More Strict"
                >
                  {result.strictness_distribution.more_strict > 0 && '↑'}
                </div>
                <div 
                  className="bg-slate-400 h-full flex items-center justify-center text-[8px] font-bold text-white transition-all"
                  style={{ width: `${(result.strictness_distribution.similar / (result.total_adaptations || 1)) * 100}%` }}
                  title="Maintained"
                >
                  {result.strictness_distribution.similar > 0 && '•'}
                </div>
                <div 
                  className="bg-emerald-500 h-full flex items-center justify-center text-[8px] font-bold text-white transition-all"
                  style={{ width: `${(result.strictness_distribution.less_strict / (result.total_adaptations || 1)) * 100}%` }}
                  title="Less Strict"
                >
                  {result.strictness_distribution.less_strict > 0 && '↓'}
                </div>
              </div>
              <div className="flex justify-between mt-2 text-[10px] font-bold text-slate-500">
                <span>{result.strictness_distribution.more_strict} STRICTOR</span>
                <span>{result.strictness_distribution.less_strict} LENIENT</span>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Recent Adaptation Events</h4>
            <div className="space-y-2">
              {result.recent_events.map((event, i) => (
                <div key={i} className="p-3 border rounded-lg bg-white flex items-center justify-between group hover:border-rose-200 transition-colors">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-xs font-bold text-slate-800">{event.gauntlet}</span>
                    <span className="text-[10px] text-slate-400 font-mono">{new Date(event.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <span className={`text-[9px] font-bold px-2 py-0.5 rounded border uppercase ${
                    event.change === 'more_strict' ? 'bg-rose-50 text-rose-600 border-rose-100' : 
                    event.change === 'less_strict' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 
                    'bg-slate-50 text-slate-600 border-slate-100'
                  }`}>
                    {event.change.replace('_', ' ')}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
