import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { PAMIResult } from '../types/plugin-types';

interface Props {
  initialTransactions?: string[][];
}

export const PAMIViz: React.FC<Props> = ({ 
  initialTransactions = [
    ['Temperature_High', 'Mutation_Fast', 'Fitness_Improved'],
    ['Temperature_Low', 'Mutation_Slow', 'Fitness_Stable'],
    ['Temperature_High', 'Mutation_Fast', 'Diversity_High'],
    ['Temperature_High', 'Fitness_Improved', 'Zero_Error']
  ]
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.pamiEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PAMIResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minSupport, setMinSupport] = useState(2);

  const runMining = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/pami/mine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          transactions: initialTransactions,
          min_support: minSupport
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Mining failed');
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
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Pattern Mining visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-end gap-4">
        <div className="flex-1 space-y-1">
          <h3 className="text-lg font-semibold text-gray-800">Frequent Pattern Discovery</h3>
          <p className="text-xs text-slate-500">Uncover hidden associations in evolutionary run data.</p>
        </div>
        <div className="w-32 space-y-1">
          <label className="text-[10px] font-bold text-slate-400 uppercase">Min Support</label>
          <input 
            type="number"
            value={minSupport}
            onChange={(e) => setMinSupport(parseInt(e.target.value))}
            min="1"
            className="w-full p-1.5 border rounded text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
          />
        </div>
        <button
          onClick={runMining}
          disabled={loading}
          className="px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors font-medium h-[38px]"
        >
          {loading ? 'Mining...' : 'Mine Patterns'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2">
          <div className="flex justify-between items-center px-1">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Mining Results (PAMI)</span>
            <span className="text-[10px] bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-bold">
              {result.total_found} Patterns
            </span>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {result.patterns.map((pattern, i) => (
              <div key={i} className="p-3 border rounded-lg bg-slate-50 flex flex-col gap-2 hover:shadow-md transition-shadow">
                <div className="flex flex-wrap gap-1">
                  {pattern.items.map((item, j) => (
                    <span key={j} className="px-2 py-0.5 bg-white border rounded text-[10px] font-mono text-indigo-600 font-bold">
                      {item}
                    </span>
                  ))}
                </div>
                <div className="flex justify-between items-center mt-1 border-t pt-2 border-slate-200">
                  <span className="text-[10px] text-slate-400 font-bold uppercase">Support Count</span>
                  <span className="text-sm font-bold text-slate-700">{pattern.support}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
