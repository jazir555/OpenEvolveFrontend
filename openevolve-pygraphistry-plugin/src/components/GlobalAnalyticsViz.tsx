import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GlobalAnalyticsResult } from '../types/plugin-types';

export const GlobalAnalyticsViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.globalAnalyticsEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<GlobalAnalyticsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/analytics/global');
      if (!response.ok) {
        throw new Error('Failed to fetch global analytics');
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
      fetchAnalytics();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Global Performance Analytics is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-slate-900 flex items-center justify-center text-white font-bold text-xs shadow-sm">A</div>
          <h3 className="text-lg font-bold text-slate-800">Global System Performance</h3>
        </div>
        <button 
          onClick={fetchAnalytics}
          disabled={loading}
          className="text-xs bg-slate-100 hover:bg-slate-200 px-2 py-1 rounded transition-colors font-bold text-slate-600"
        >
          {loading ? 'Aggregating...' : 'Refresh Summary'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-900 text-white rounded-xl shadow-lg">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Total Cumulative Cost</p>
              <p className="text-3xl font-mono font-bold text-emerald-400 mt-1">${result.total_cost.toFixed(2)}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Global Token Usage</p>
              <p className="text-2xl font-bold text-slate-800 mt-1">{result.total_tokens.toLocaleString()}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Workflows Tracked</p>
              <p className="text-2xl font-bold text-slate-800 mt-1">{result.total_workflows}</p>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Provider Cost Breakdown</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(result.provider_breakdown).map(([provider, metrics]) => (
                <div key={provider} className="p-3 border rounded-lg bg-white shadow-sm flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-bold text-slate-700 capitalize">{provider}</span>
                    <span className="text-xs font-mono font-bold text-emerald-600">${metrics.cost.toFixed(4)}</span>
                  </div>
                  <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                    <div 
                      className="bg-indigo-500 h-full" 
                      style={{ width: `${Math.min(100, (metrics.cost / result.total_cost) * 100)}%` }} 
                    />
                  </div>
                  <p className="text-[10px] text-slate-400 font-medium">
                    {metrics.tokens.toLocaleString()} tokens utilized
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
