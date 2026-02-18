import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GenericResult } from '../types/plugin-types';

export const OrchestratorViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const [plannerData, setPlannerData] = useState<GenericResult | null>(null);
  const [crewaiData, setCrewAIData] = useState<GenericResult | null>(null);
  const [romaData, setRomaData] = useState<GenericResult | null>(null);

  const fetchData = async () => {
    if (state.features.e2ePlannerEnabled) {
      const res = await fetch('/api/openevolve/planner/e2e');
      setPlannerData(await res.json());
    }
    if (state.features.crewaiEnabled) {
      const res = await fetch('/api/openevolve/crewai/summary');
      setCrewAIData(await res.json());
    }
    if (state.features.romaEnabled) {
      const res = await fetch('/api/openevolve/roma/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: "Recursive synthesis of 100+ components" })
      });
      setRomaData(await res.json());
    }
  };

  useEffect(() => {
    fetchData();
  }, [state.features.e2ePlannerEnabled, state.features.crewaiEnabled, state.features.romaEnabled]);

  return (
    <div className="p-4 flex flex-col gap-6">
      {state.features.romaEnabled && romaData && (
        <div className="p-4 border rounded-xl bg-slate-50 border-indigo-200">
          <h3 className="text-lg font-bold text-slate-800 mb-2">ROMA Recursive Meta-Agent</h3>
          <p className="text-[10px] text-slate-500 uppercase font-bold mb-3 tracking-widest">Synthesized Response</p>
          <div className="p-3 bg-white border rounded shadow-inner text-xs text-slate-600 leading-relaxed italic">
            {romaData.synthesized_result}
          </div>
        </div>
      )}

      {state.features.crewaiEnabled && crewaiData && (
        <div className="p-4 border rounded-xl bg-white shadow-sm">
          <h3 className="text-lg font-bold text-slate-800 mb-4">CrewAI Workflow Status</h3>
          <div className="flex gap-4">
            {Object.entries(crewaiData.status_distribution || {}).map(([status, count]) => (
              <div key={status} className="flex-1 p-2 bg-slate-50 rounded border text-center">
                <p className="text-[8px] font-bold text-slate-400 uppercase">{status}</p>
                <p className="text-xl font-black text-indigo-600">{count as number}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      {state.features.e2ePlannerEnabled && plannerData && (
        <div className="p-6 border rounded-2xl bg-white shadow-xl border-indigo-100">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-black text-slate-900 tracking-tight">E2E Invention Planner</h3>
            <span className="px-3 py-1 bg-indigo-600 text-white rounded-full text-xs font-bold uppercase tracking-widest animate-pulse">
              Orchestrating
            </span>
          </div>

          <div className="relative h-4 w-full bg-slate-100 rounded-full overflow-hidden mb-8 shadow-inner">
            <div className="absolute top-0 left-0 h-full bg-indigo-500 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(99,102,241,0.5)]" 
                 style={{ width: `${(plannerData.completion || 0) * 100}%` }} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {(plannerData.milestones || []).map((m: string) => (
              <div key={m} className={`p-4 rounded-xl border transition-all ${
                plannerData.current === m ? 'bg-indigo-50 border-indigo-200 shadow-sm scale-105' : 'bg-slate-50 border-slate-100 opacity-50'
              }`}>
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Milestone</p>
                <p className={`text-sm font-bold ${plannerData.current === m ? 'text-indigo-700' : 'text-slate-600'}`}>{m}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
