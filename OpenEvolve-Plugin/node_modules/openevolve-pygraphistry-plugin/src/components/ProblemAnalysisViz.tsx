import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ProblemAnalysisResult } from '../types/plugin-types';

interface Props {
  initialText?: string;
}

export const ProblemAnalysisViz: React.FC<Props> = ({ 
  initialText = ''
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.problemAnalysisEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ProblemAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState(initialText);

  const runAnalysis = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/problem/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problem_text: text })
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
        <p className="font-medium italic">Problem Analysis visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="space-y-2">
        <h3 className="text-lg font-bold text-slate-800">Semantic Problem Analysis</h3>
        <textarea 
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe the complex problem to analyze..."
          className="w-full p-3 border rounded-md min-h-[100px] focus:ring-2 focus:ring-indigo-500 outline-none text-sm"
        />
        <div className="flex justify-end">
          <button
            onClick={runAnalysis}
            disabled={loading || !text}
            className="px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors font-bold text-sm"
          >
            {loading ? 'Analyzing...' : 'Perform Analysis'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4">
          <div className="flex justify-between items-start">
            <div>
              <h4 className="text-xl font-bold text-slate-900">{result.title}</h4>
              <p className="text-xs text-slate-500 mt-1 uppercase font-bold tracking-widest">{result.domain} • {result.problem_type}</p>
            </div>
            <div className="text-right">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Overall Complexity</span>
              <p className="text-2xl font-mono font-bold text-indigo-600">{result.complexity.overall.toFixed(1)}/10</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h5 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Identified Constraints</h5>
              <div className="space-y-2">
                {result.constraints.map((c, i) => (
                  <div key={i} className="p-2 bg-slate-50 border rounded-lg flex flex-col gap-1">
                    <div className="flex justify-between">
                      <span className="text-[8px] font-bold uppercase text-indigo-500">{c.type}</span>
                      <span className={`text-[8px] font-bold uppercase ${c.severity === 'hard' ? 'text-rose-500' : 'text-amber-500'}`}>
                        {c.severity}
                      </span>
                    </div>
                    <p className="text-xs text-slate-700">{c.description}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <h5 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Success Criteria</h5>
              <div className="space-y-2">
                {result.success_criteria.map((sc, i) => (
                  <div key={i} className="p-2 border border-emerald-100 bg-emerald-50/30 rounded-lg">
                    <p className="text-xs font-bold text-emerald-800">{sc.description}</p>
                    <div className="flex justify-between mt-1">
                      <span className="text-[10px] text-emerald-600/70 font-medium">{sc.metric}</span>
                      <span className="text-[10px] font-bold text-emerald-700">Threshold: {sc.threshold}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
