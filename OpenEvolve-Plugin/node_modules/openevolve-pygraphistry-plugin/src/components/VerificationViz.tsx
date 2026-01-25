import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { VerificationResult } from '../types/plugin-types';

export const VerificationViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.verificationEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runVerification = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/verification/run');
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Verification failed');
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
        <p className="font-medium italic">Mathematical Verification visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Algorithmic Correctness Analysis</h3>
          <p className="text-xs text-slate-500">Formal mathematical verification of core system algorithms.</p>
        </div>
        <button
          onClick={runVerification}
          disabled={loading}
          className="px-4 py-2 bg-slate-900 text-white rounded hover:bg-black disabled:opacity-50 transition-colors font-bold text-sm"
        >
          {loading ? 'Verifying...' : 'Run Analysis'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="flex items-center gap-4 p-4 bg-slate-900 text-white rounded-xl shadow-lg border border-slate-800">
            <div className={`w-16 h-16 rounded-full border-4 flex items-center justify-center text-xl font-bold ${result.success_rate === 1.0 ? 'border-emerald-500 text-emerald-400' : 'border-rose-500 text-rose-400'}`}>
              {(result.success_rate * 100).toFixed(0)}%
            </div>
            <div>
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Verification Summary</p>
              <p className="text-lg font-medium">{result.passed} / {result.total_tests} algorithmic properties verified</p>
              <p className="text-xs text-slate-500 mt-1">Last run: {new Date(result.timestamp).toLocaleString()}</p>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Proof Logs</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {result.results.map((test, i) => (
                <div key={i} className="p-3 border rounded-lg bg-slate-50 flex items-center justify-between group hover:bg-white hover:border-slate-300 transition-all">
                  <div className="space-y-0.5">
                    <p className="text-[10px] font-bold text-slate-400 uppercase leading-none">{test.category}</p>
                    <p className="text-sm font-medium text-slate-700">{test.test}</p>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                    test.status === 'passed' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 'bg-rose-50 text-rose-600 border-rose-100'
                  }`}>
                    {test.status.toUpperCase()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
