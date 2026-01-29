import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { LeanAideResult } from '../types/plugin-types';

interface Props {
  initialTheorem?: string;
}

export const LeanAideViz: React.FC<Props> = ({ 
  initialTheorem = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.leanAideEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<LeanAideResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [theorem, setTheorem] = useState(initialTheorem);

  const runAutoformalization = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/leanaide/formalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theorem_text: theorem })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Formalization failed');
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
      <div className="p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400">
        <p className="font-medium italic">LeanAide Autoformalization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-800">Mathematical Autoformalization</h3>
        <p className="text-xs text-slate-500">Translate natural language math to Lean4 formal proofs.</p>
        <textarea 
          value={theorem}
          onChange={(e) => setTheorem(e.target.value)}
          placeholder="Enter a mathematical theorem (e.g., 'There are infinitely many primes')..."
          className="w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-emerald-500 outline-none text-sm font-sans"
        />
        <div className="flex justify-end">
          <button
            onClick={runAutoformalization}
            disabled={loading || !theorem}
            className="px-6 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors font-medium"
          >
            {loading ? 'Formalizing...' : 'Formalize to Lean4'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-top-2">
          <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 overflow-x-auto">
            <div className="flex justify-between items-center mb-2 border-b border-slate-700 pb-2">
              <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">Lean4 Output</span>
              <span className="text-[10px] text-slate-500 font-mono">Confidence: {(result.confidence * 100).toFixed(0)}%</span>
            </div>
            <pre className="text-xs font-mono text-slate-200 leading-relaxed">
              {result.theorem_lean}
            </pre>
          </div>
          
          <div className="flex items-center gap-2 px-1">
            <div className={`w-2 h-2 rounded-full ${result.proof_status === 'verified' ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            <span className="text-xs font-medium text-slate-600 uppercase tracking-tight">Status: {result.proof_status}</span>
          </div>
        </div>
      )}
    </div>
  );
};
