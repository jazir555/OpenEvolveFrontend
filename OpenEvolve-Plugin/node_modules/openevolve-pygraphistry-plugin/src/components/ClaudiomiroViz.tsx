import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ClaudiomiroResult } from '../types/plugin-types';

interface Props {
  initialPrompt?: string;
  workingDir?: string;
}

export const ClaudiomiroViz: React.FC<Props> = ({ 
  initialPrompt = '',
  workingDir = '.'
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.claudiomiroEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ClaudiomiroResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(initialPrompt);

  const runDecomposition = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/claudiomiro/decompose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, working_dir: workingDir })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Decomposition failed');
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
        <p className="font-medium italic">Claudiomiro Autonomous Dev visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-slate-700 flex items-center justify-center text-white font-bold text-xs shadow-sm">C</div>
          <h3 className="text-lg font-bold text-slate-800">Autonomous Development (Claudiomiro)</h3>
        </div>
        <p className="text-xs text-slate-500">Autonomous task decomposition and parallel sub-task generation.</p>
        <textarea 
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a development task (e.g., 'Refactor the data ingestion pipeline to support Parquet format')..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-slate-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runDecomposition}
            disabled={loading || !prompt}
            className="px-6 py-2 bg-slate-800 text-white rounded hover:bg-slate-900 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
          >
            {loading ? 'Decomposing...' : 'Decompose Task'}
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
            <span className="text-[10px] font-bold text-slate-700 uppercase tracking-widest bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              Task Breakdown: {result.task_id}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">{result.num_tasks} Sub-tasks</span>
          </div>
          
          <div className="space-y-2">
            {result.sub_tasks.map((task, i) => (
              <div key={i} className="p-3 bg-slate-50 border rounded-lg flex flex-col gap-1 hover:bg-white hover:border-slate-300 transition-all group">
                <div className="flex justify-between items-center">
                  <h4 className="text-sm font-bold text-slate-800">{task.title}</h4>
                  <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${
                    task.status === 'completed' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 'bg-amber-50 text-amber-600 border-amber-100'
                  }`}>
                    {task.status}
                  </span>
                </div>
                <p className="text-xs text-slate-500 line-clamp-2">{task.description}</p>
              </div>
            ))}
          </div>

          <div className="flex items-start gap-3 p-3 bg-slate-50/50 rounded border border-slate-100/50">
            <span className="text-lg">🛠️</span>
            <p className="text-[11px] text-slate-600 leading-snug italic">
              <strong>Claudiomiro</strong> has mapped these sub-tasks to a parallel execution DAG. 
              In full autonomous mode, each would be resolved with automated testing and commits.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
