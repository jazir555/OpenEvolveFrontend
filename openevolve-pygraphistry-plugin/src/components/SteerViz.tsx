import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { SteerResult } from '../types/plugin-types';

interface Props {
  task: string;
  output: string;
}

export const SteerViz: React.FC<Props> = ({ 
  task,
  output
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.steerEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<SteerResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runVerification = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/steer/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, output })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Reliability verification failed');
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
        <p className="font-medium italic">Active Reliability (Steer) visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-blue-900 flex items-center justify-center text-white font-bold text-xs shadow-sm">S</div>
          <h3 className="text-lg font-bold text-slate-800">Active Reliability (ACE + Steer)</h3>
        </div>
        <button
          onClick={runVerification}
          disabled={loading || !output}
          className="px-4 py-2 bg-blue-900 text-white rounded hover:bg-blue-950 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm"
        >
          {loading ? 'Verifying...' : 'Verify Output'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="flex items-center gap-3 p-3 rounded-lg border bg-slate-50">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center text-xl ${result.all_passed ? 'bg-emerald-100 text-emerald-600' : 'bg-rose-100 text-rose-600'}`}>
              {result.all_passed ? '🔒' : '🔓'}
            </div>
            <div>
              <p className="text-sm font-bold text-slate-800">Reality Lock Status: {result.all_passed ? 'LOCKED' : 'UNLOCKED'}</p>
              <p className="text-xs text-slate-500">{result.all_passed ? 'Deterministic quality standards met.' : 'Verification failed. Closed-loop learning triggered.'}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Judge Results</h4>
              {result.results.map((r, i) => (
                <div key={i} className="p-2 border rounded bg-white flex justify-between items-center">
                  <span className="text-xs font-medium text-slate-700">{r.judge}</span>
                  <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border ${r.passed ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 'bg-rose-50 text-rose-600 border-rose-100'}`}>
                    {r.passed ? 'PASSED' : 'FAILED'}
                  </span>
                </div>
              ))}
            </div>

            {result.ace_learning && (
              <div className="space-y-2">
                <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest px-1">Closed-Loop Learning</h4>
                <div className="p-3 border border-indigo-100 bg-indigo-50/30 rounded-lg">
                  <p className="text-[10px] font-bold text-indigo-700 uppercase mb-1">Skills Acquired:</p>
                  <div className="flex flex-wrap gap-1">
                    {result.ace_learning.learned_skills.map(skill => (
                      <span key={skill} className="px-2 py-0.5 bg-white border border-indigo-200 rounded text-[10px] text-indigo-600 font-medium">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
