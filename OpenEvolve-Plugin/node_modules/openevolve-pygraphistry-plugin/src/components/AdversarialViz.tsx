import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { AdversarialResult as IAdversarialResult } from '../types/plugin-types';

interface Props {
  content: string;
  theorem?: string;
}

export const AdversarialViz: React.FC<Props> = ({ 
  content,
  theorem = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.adversarialEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<IAdversarialResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAdversarialTest = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/adversarial/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, theorem })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Adversarial validation failed');
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
      <div className="p-6 text-center border-2 border-dashed border-red-100 rounded-lg bg-red-50/30 text-red-400">
        <p className="font-medium italic">Adversarial Validation is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">Red/Blue Team Validation</h3>
          <p className="text-xs text-slate-500">Adversarial stress-testing for proof robustness.</p>
        </div>
        <button
          onClick={runAdversarialTest}
          disabled={loading || !content}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 transition-colors font-medium shadow-sm"
        >
          {loading ? 'Attacking...' : 'Run Stress Test'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-900 text-white rounded-lg border border-slate-800 flex flex-col items-center justify-center">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Robustness Score</span>
              <span className={`text-3xl font-mono mt-1 ${result.is_robust ? 'text-emerald-400' : 'text-rose-400'}`}>
                {(result.robustness_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg border flex flex-col items-center justify-center">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Attacks Blocked</span>
              <span className="text-3xl font-mono mt-1 text-slate-700">
                {result.attacks_blocked} / {result.total_attacks}
              </span>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg border flex flex-col items-center justify-center">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Status</span>
              <span className={`text-lg font-bold mt-1 uppercase ${result.is_robust ? 'text-emerald-600' : 'text-rose-600'}`}>
                {result.is_robust ? '🛡️ Robust' : '⚠️ Vulnerable'}
              </span>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Red Team Attack Logs</h4>
            <div className="space-y-2">
              {result.attack_results.map((attack, i) => (
                <div key={i} className={`p-3 border rounded-lg flex items-center gap-4 transition-all ${attack.success ? 'bg-rose-50 border-rose-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  <div className={`flex-none w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg ${attack.success ? 'bg-rose-200 text-rose-700' : 'bg-emerald-200 text-emerald-700'}`}>
                    {attack.success ? '💥' : '🛡️'}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs font-bold uppercase tracking-tight text-slate-600">{attack.strategy} Attack</span>
                      <span className="text-[10px] font-mono bg-white px-1.5 rounded border">Severity: {attack.severity.toFixed(2)}</span>
                    </div>
                    <p className="text-sm text-slate-700">{attack.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
