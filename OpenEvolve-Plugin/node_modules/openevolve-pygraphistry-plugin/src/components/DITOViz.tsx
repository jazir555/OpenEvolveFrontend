import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { DITOResult } from '../types/plugin-types';

export const DITOViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.ditoEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DITOResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/constraints/dito');
      if (!response.ok) {
        throw new Error('Failed to run DITO analysis');
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
        <p className="font-medium italic">High-Performance Logic Audit is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-indigo-900 to-slate-900 flex items-center justify-center text-white font-bold text-xs shadow-md">D</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Logic Contradiction Audit (DITO)</h3>
        </div>
        <button 
          onClick={runAnalysis}
          disabled={loading}
          className="px-4 py-2 bg-slate-900 text-white rounded hover:bg-black disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
        >
          {loading ? 'Analyzing...' : 'Run Audit'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Index Size</p>
              <p className="text-2xl font-mono font-bold text-slate-800">{result.total_constraints} Nodes</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Query Complexity</p>
              <p className="text-2xl font-mono font-bold text-indigo-600">O(log n)</p>
            </div>
            <div className={`p-4 rounded-xl border text-center ${result.contradiction_count > 0 ? 'bg-rose-50 border-rose-100' : 'bg-emerald-50 border-emerald-100'}`}>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Conflicts Found</p>
              <p className={`text-2xl font-mono font-bold ${result.contradiction_count > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                {result.contradiction_count}
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Detected Logical Collisions</h4>
            <div className="space-y-2">
              {result.contradictions.map((c) => (
                <div key={c.id} className="p-3 border border-rose-100 bg-rose-50/20 rounded-lg group hover:bg-rose-50 transition-colors">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[10px] font-bold text-rose-700 uppercase">{c.pair.join(' ↔ ')}</span>
                    <span className="text-[10px] font-mono font-bold text-rose-400">Confidence: {c.confidence.toFixed(2)}</span>
                  </div>
                  <p className="text-xs text-slate-700 font-medium leading-relaxed">{c.description}</p>
                </div>
              ))}
              
              {result.contradiction_count === 0 && (
                <div className="py-8 text-center text-slate-400 border-2 border-dashed rounded-lg">
                  No logical contradictions detected by spatial hashing.
                </div>
              )}
            </div>
          </div>

          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-[10px] font-mono text-indigo-400/60">
              <span>// DITO: Dynamic Inference Trace Optimizer</span>
              <span>v1.0.0-stable</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
