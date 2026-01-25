import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { Lean4Theorem } from '../types/plugin-types';

export const Lean4Viz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.lean4Enabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [theorems, setTheorems] = useState<Lean4Theorem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchTheorems = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/mathematics/lean4');
      if (!response.ok) {
        throw new Error('Failed to fetch Lean 4 theorems');
      }
      const data = await response.json();
      setTheorems(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchTheorems();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400">
        <p className="font-medium italic">Lean 4 Formal Verification visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-white font-bold text-xs shadow-sm">L</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Lean 4 Theorem Prover</h3>
        </div>
        <button 
          onClick={fetchTheorems}
          disabled={loading}
          className="text-xs font-bold text-indigo-600 hover:underline px-2 py-1"
        >
          {loading ? 'Refreshing...' : 'Sync Theorems'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="space-y-4 animate-in fade-in slide-in-from-top-2">
        {theorems.map((t) => (
          <div key={t.name} className="p-4 border rounded-xl bg-slate-50 border-l-4 border-l-slate-800">
            <div className="flex justify-between items-start mb-3">
              <h4 className="text-sm font-mono font-bold text-slate-900">{t.name}</h4>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                t.verified ? 'bg-emerald-50 text-emerald-600 border-emerald-200' : 'bg-amber-50 text-amber-600 border-amber-200'
              }`}>
                {t.verified ? '✓ VERIFIED' : '○ UNVERIFIED'}
              </span>
            </div>
            
            <div className="space-y-3">
              <div className="p-3 bg-white border rounded-lg shadow-inner">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Statement</p>
                <code className="text-xs text-indigo-700 font-mono leading-relaxed">{t.statement}</code>
              </div>
              
              <div className="p-3 bg-slate-900 rounded-lg shadow-inner group">
                <div className="flex justify-between items-center mb-1">
                  <p className="text-[10px] font-bold text-slate-500 uppercase">Proof Sketch</p>
                  <span className="text-[8px] font-bold text-slate-600 uppercase opacity-0 group-hover:opacity-100 transition-opacity">Tactics Mode</span>
                </div>
                <pre className="text-xs text-slate-300 font-mono overflow-x-auto">
                  {t.proof_sketch}
                </pre>
              </div>
            </div>
          </div>
        ))}

        {!loading && theorems.length === 0 && !error && (
          <div className="py-12 text-center text-slate-400 border-2 border-dashed rounded-lg font-medium italic">
            No Lean 4 theorems discovered in context.
          </div>
        )}
      </div>
    </div>
  );
};
