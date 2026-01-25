import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { DataPizzaResult } from '../types/plugin-types';

interface Props {
  initialTask?: string;
}

export const DataPizzaViz: React.FC<Props> = ({ 
  initialTask = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.datapizzaEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DataPizzaResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState(initialTask);

  const runDataPizza = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/datapizza/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Multi-agent execution failed');
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
      <div className="p-6 text-center border-2 border-dashed border-rose-100 rounded-lg bg-rose-50/30 text-rose-400">
        <p className="font-medium italic">DataPizza Multi-Agent visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-rose-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">DP</div>
          <h3 className="text-lg font-bold text-slate-800">Multi-Agent Data Processing (DataPizza)</h3>
        </div>
        <textarea 
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Enter a task for the multi-agent team (e.g., 'Analyze the security of our data transformation logic')..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-rose-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runDataPizza}
            disabled={loading || !task}
            className="px-6 py-2 bg-rose-600 text-white rounded hover:bg-rose-700 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
          >
            {loading ? 'Processing...' : 'Run Team Workflow'}
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
            <span className="text-[10px] font-bold text-rose-700 uppercase tracking-widest bg-rose-50 px-2 py-0.5 rounded border border-rose-100">
              Team: {result.team_name}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">Total Steps: {result.total_steps}</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Blue Team (Solver) */}
            {result.results.blue && (
              <div className="p-3 border rounded-lg bg-blue-50/50 border-blue-100">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-4 h-4 rounded-full bg-blue-500 block" />
                  <span className="text-[10px] font-bold text-blue-700 uppercase">Blue Team (Solver)</span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed">{result.results.blue.response}</p>
              </div>
            )}

            {/* Red Team (Critiquer) */}
            {result.results.red && (
              <div className="p-3 border rounded-lg bg-rose-50/50 border-rose-100">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-4 h-4 rounded-full bg-rose-500 block" />
                  <span className="text-[10px] font-bold text-rose-700 uppercase">Red Team (Critiquer)</span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed">{result.results.red.response}</p>
              </div>
            )}

            {/* Gold Team (Verifier) */}
            {result.results.gold && (
              <div className="p-3 border rounded-lg bg-amber-50/50 border-amber-100">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-4 h-4 rounded-full bg-amber-500 block" />
                  <span className="text-[10px] font-bold text-amber-700 uppercase">Gold Team (Verifier)</span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed">{result.results.gold.response}</p>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 px-1">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tight">Status: {result.status}</span>
          </div>
        </div>
      )}
    </div>
  );
};
