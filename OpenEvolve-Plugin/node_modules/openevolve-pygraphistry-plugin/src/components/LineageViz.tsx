import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { LineageTrace } from '../types/plugin-types';

export const LineageViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.lineageEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [traces, setTraces] = useState<LineageTrace[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchLineage = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/evolution/lineage');
      if (!response.ok) {
        throw new Error('Failed to fetch lineage traces');
      }
      const data = await response.json();
      setTraces(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchLineage();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Evolution Lineage visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Evolution Ancestry & Lineage</h3>
          <p className="text-xs text-slate-500 font-medium">Ancestral graph of program improvements and generation depth.</p>
        </div>
        <button 
          onClick={fetchLineage}
          disabled={loading}
          className="text-xs font-bold text-indigo-600 hover:underline px-2 py-1"
        >
          {loading ? 'Tracing...' : 'Extract Lineage'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
        {traces.map((trace) => (
          <div key={trace.final_program_id} className="p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-sm font-bold text-slate-800">Final Candidate: {trace.final_program_id}</h4>
              <span className="text-[10px] font-bold bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full border border-indigo-200 uppercase">
                Depth: {trace.generation_depth} Generations
              </span>
            </div>

            <div className="relative">
              <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-200" />
              
              <div className="space-y-4 relative">
                {trace.improvement_steps.map((step, i) => (
                  <div key={i} className="flex gap-4 items-start ml-2">
                    <div className="w-4 h-4 rounded-full bg-white border-2 border-indigo-500 flex-none z-10 mt-1" />
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 uppercase mb-1">
                        <span>Step {step.step}: {step.parent_id} → {step.child_id}</span>
                        {step.generation && <span>Gen {step.generation}</span>}
                      </div>
                      <div className="bg-white border rounded-lg p-2 shadow-sm">
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(step.improvement).map(([metric, delta]) => (
                            <div key={metric} className="flex items-center gap-1.5 px-1.5 py-0.5 bg-emerald-50 border border-emerald-100 rounded text-[9px] font-bold text-emerald-700">
                              <span className="uppercase opacity-60">{metric}</span>
                              <span>+{delta.toFixed(4)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}

        {!loading && traces.length === 0 && !error && (
          <div className="py-12 text-center text-slate-400 border-2 border-dashed rounded-lg">
            No lineage traces found in checkpoints.
          </div>
        )}
      </div>
    </div>
  );
};
