import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { StaticAnalysisResult } from '../types/plugin-types';

export const StaticAnalysisViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.staticAnalysisEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<StaticAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/analysis/static', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_paths: [] }) // Default to core files
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
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
        <p className="font-medium italic">Static Code Analysis is currently disabled in settings.</p>
      </div>
    );
  }

  const getSeverityColor = (sev: string) => {
    switch (sev.toLowerCase()) {
      case 'critical': return 'text-rose-600 bg-rose-50 border-rose-100';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-100';
      case 'medium': return 'text-amber-600 bg-amber-50 border-amber-100';
      default: return 'text-slate-600 bg-slate-50 border-slate-100';
    }
  };

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Deep Static Code Analysis</h3>
          <p className="text-xs text-slate-500 font-medium">Security vulnerability & code quality scanning.</p>
        </div>
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="px-4 py-2 bg-slate-800 text-white rounded hover:bg-black disabled:opacity-50 transition-colors font-bold text-sm"
        >
          {loading ? 'Scanning...' : 'Run Full Scan'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-slate-900 text-white rounded-xl shadow-md border border-slate-800 text-center">
              <p className="text-[10px] font-bold text-slate-400 uppercase">Total Issues</p>
              <p className="text-2xl font-mono font-bold text-emerald-400">{result.summary.total_issues}</p>
            </div>
            {Object.entries(result.summary.by_severity).map(([sev, count]) => (
              <div key={sev} className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-center">
                <p className="text-[10px] font-bold text-slate-400 uppercase">{sev}</p>
                <p className="text-xl font-bold text-slate-700">{count}</p>
              </div>
            ))}
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Top Security & Quality Risks</h4>
            <div className="space-y-2">
              {Object.entries(result.issues_by_severity).flatMap(([sev, issues]) => 
                issues.map((issue, i) => (
                  <div key={`${sev}-${i}`} className={`p-3 border rounded-lg flex flex-col gap-1 hover:shadow-sm transition-all border-l-4 ${getSeverityColor(sev).split(' ')[2]}`}>
                    <div className="flex justify-between items-center">
                      <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${getSeverityColor(sev)}`}>
                        {sev}
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">{issue.file}:{issue.line}</span>
                    </div>
                    <p className="text-sm font-medium text-slate-800">{issue.message}</p>
                    {issue.suggestion && (
                      <p className="text-[11px] text-slate-500 italic mt-1 bg-slate-50/50 p-1 rounded border border-dashed">
                        💡 Suggestion: {issue.suggestion}
                      </p>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
