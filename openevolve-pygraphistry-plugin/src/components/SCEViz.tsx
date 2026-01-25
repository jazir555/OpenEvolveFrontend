import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { SymbolicConstraint } from '../types/plugin-types';

export const SCEViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.sceEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [constraints, setConstraints] = useState<SymbolicConstraint[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchConstraints = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/constraints/symbolic');
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch symbolic constraints');
      }
      const data = await response.json();
      setConstraints(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchConstraints();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Symbolic Logic visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Symbolic Constraint Engine (SCE)</h3>
          <p className="text-xs text-slate-500 font-medium">Formal logical constraints and proof status.</p>
        </div>
        <button 
          onClick={fetchConstraints}
          disabled={loading}
          className="text-xs font-bold text-indigo-600 hover:underline px-2 py-1"
        >
          {loading ? 'Fetching...' : 'Sync Constraints'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="space-y-3 animate-in fade-in slide-in-from-top-2">
        {constraints.map((c) => (
          <div key={c.id} className="p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all border-l-4 border-l-indigo-500">
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${
                  c.type === 'hard' ? 'bg-rose-50 text-rose-700 border-rose-100' : 'bg-blue-50 text-blue-700 border-blue-100'
                }`}>
                  {c.type}
                </span>
                <h4 className="text-sm font-bold text-slate-800">{c.id}</h4>
              </div>
              <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase border ${
                c.verified ? 'bg-emerald-50 text-emerald-600 border-emerald-200' : 'bg-amber-50 text-amber-600 border-amber-200'
              }`}>
                {c.verified ? '✓ Verified (Lean4)' : '○ Pending Proof'}
              </div>
            </div>
            
            <p className="text-sm text-slate-600 mb-3">{c.description}</p>
            
            <div className="p-2 bg-slate-900 rounded border border-slate-800 overflow-x-auto">
              <code className="text-[11px] font-mono text-indigo-300">
                {c.formalization}
              </code>
            </div>
            
            <div className="mt-2 flex justify-end">
              <span className="text-[9px] text-slate-400 font-medium">Source: {c.source}</span>
            </div>
          </div>
        ))}

        {!loading && constraints.length === 0 && !error && (
          <div className="py-12 text-center text-slate-400 border-2 border-dashed rounded-lg">
            No symbolic constraints found.
          </div>
        )}
      </div>
    </div>
  );
};
