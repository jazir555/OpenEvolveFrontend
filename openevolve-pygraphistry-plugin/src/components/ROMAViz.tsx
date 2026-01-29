import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ROMAResult } from '../types/plugin-types';

interface Props {
  initialTask?: string;
}

export const ROMAViz: React.FC<Props> = ({ 
  initialTask = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.romaEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ROMAResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState(initialTask);

  const runROMA = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/roma/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Recursive solving failed');
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
      <div className="p-6 text-center border-2 border-dashed border-cyan-100 rounded-lg bg-cyan-50/30 text-cyan-400">
        <p className="font-medium italic">ROMA Orchestration visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-cyan-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">R</div>
          <h3 className="text-lg font-bold text-slate-800">Recursive Meta-Agents (ROMA)</h3>
        </div>
        <p className="text-xs text-slate-500">Hierarchical task decomposition and recursive agent orchestration.</p>
        <textarea 
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Describe a complex multi-step task..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-cyan-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runROMA}
            disabled={loading || !task}
            className="px-6 py-2 bg-cyan-700 text-white rounded hover:bg-cyan-800 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
          >
            {loading ? 'Orchestrating...' : 'Solve Recursively'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-cyan-700 uppercase tracking-widest bg-cyan-50 px-2 py-0.5 rounded border border-cyan-100">
              Synthesized Solution
            </span>
            <span className="text-[10px] text-slate-400 font-mono">Status: {result.status}</span>
          </div>
          
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 shadow-inner">
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap font-medium">
              {result.synthesized_result}
            </p>
          </div>

          <div className="flex items-start gap-3 p-3 bg-cyan-50/50 rounded border border-cyan-100/50">
            <span className="text-lg">🤖</span>
            <p className="text-[11px] text-cyan-800/80 leading-snug">
              <strong>Engine Note:</strong> ROMA used recursive planning to break this into atomic subtasks, 
              invoking specialized executors for each, and aggregating them into this final result.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
