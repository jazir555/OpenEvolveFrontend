import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { LossMapping } from '../types/plugin-types';

export const LLTLViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.lltlEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [mappings, setMappings] = useState<LossMapping[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchMappings = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/constraints/loss-mapping');
      if (!response.ok) {
        throw new Error('Failed to fetch loss mappings');
      }
      const data = await response.json();
      setMappings(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchMappings();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-violet-100 rounded-lg bg-violet-50/30 text-violet-400">
        <p className="font-medium italic">Logic-to-Loss visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Logic-to-Loss Translation (LLTL)</h3>
          <p className="text-xs text-slate-500">Mapping symbolic constraints to differentiable loss functions.</p>
        </div>
        <button 
          onClick={fetchMappings}
          disabled={loading}
          className="text-xs font-bold text-violet-600 hover:underline"
        >
          {loading ? 'Translating...' : 'Refresh Mappings'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 animate-in fade-in slide-in-from-top-2">
        {mappings.map((m) => (
          <div key={m.constraint_id} className="p-4 border rounded-xl bg-slate-50 relative group overflow-hidden">
            <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
              <span className="text-4xl font-bold">∫</span>
            </div>
            
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">{m.constraint_id}</h4>
            
            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <div>
                  <p className="text-[10px] text-slate-400 font-bold uppercase">Relaxation</p>
                  <p className="text-sm font-mono font-bold text-violet-600 capitalize">{m.fuzzy_type}</p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-slate-400 font-bold uppercase">Weight</p>
                  <p className="text-lg font-mono font-bold text-slate-800">{m.weight.toFixed(1)}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${m.success ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                  {m.success ? 'Differentiable' : 'Translation Failed'}
                </span>
              </div>
            </div>

            {m.error && (
              <p className="mt-2 text-[10px] text-rose-600 font-medium bg-rose-50 p-1 rounded">{m.error}</p>
            )}
          </div>
        ))}
      </div>

      <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 mt-4">
        <p className="text-[10px] text-slate-400 font-mono italic">
          // Foundation: LLTL enables backpropagation through formal logical constraints 
          // by relaxing discrete propositions into smooth barrier and penalty functions.
        </p>
      </div>
    </div>
  );
};
