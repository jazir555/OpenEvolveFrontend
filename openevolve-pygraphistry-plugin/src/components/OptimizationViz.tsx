import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';

interface Props {
  problemType: string;
  initialValue?: number;
}

export const OptimizationViz: React.FC<Props> = ({ 
  problemType,
  initialValue = 10.0
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.optimizationEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [history, setHistory] = useState<number[]>([]);
  const [optimalValue, setOptimalValue] = useState<number | null>(null);

  const startOptimization = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/openevolve/optimization/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          problem_type: problemType,
          initial_value: initialValue
        })
      });

      if (!response.ok) {
        throw new Error('Optimization failed');
      }

      const data = await response.json();
      setHistory(data.convergence || []);
      setOptimalValue(data.optimal_value);
    } catch (err) {
      console.error('Optimization failed:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400">
        <p className="font-medium italic">NeuroMANCER Optimization visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="p-4 border rounded-lg bg-slate-50 space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800">NeuroMANCER Optimization</h3>
          <p className="text-xs text-slate-500">Problem: {problemType}</p>
        </div>
        <button
          onClick={startOptimization}
          disabled={loading}
          className="px-4 py-2 bg-emerald-600 text-white rounded shadow hover:bg-emerald-700 disabled:opacity-50"
        >
          {loading ? 'Solving...' : 'Run Optimization'}
        </button>
      </div>

      {optimalValue !== null && (
        <div className="p-3 bg-white rounded border border-emerald-100 flex justify-around">
          <div className="text-center">
            <p className="text-xs text-slate-400 uppercase">Optimal Value</p>
            <p className="text-xl font-mono text-emerald-600">{optimalValue.toFixed(4)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-slate-400 uppercase">Iterations</p>
            <p className="text-xl font-mono text-slate-700">{history.length}</p>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="h-40 flex items-end space-x-1 border-b border-l p-2 bg-white">
          {history.map((val, i) => (
            <div 
              key={i}
              className="bg-emerald-400 w-full hover:bg-emerald-500 transition-all"
              style={{ height: `${(val / initialValue) * 100}%` }}
              title={`Step ${i}: ${val.toFixed(2)}`}
            />
          ))}
        </div>
      )}
      
      <p className="text-[10px] text-slate-400 text-center italic">
        NeuroMANCER: Differentiable Programming with Physics Constraints
      </p>
    </div>
  );
};
