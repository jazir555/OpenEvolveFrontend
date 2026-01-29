import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ResearchStage } from '../types/plugin-types';

export const ResearchQuestViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.researchQuestEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [stages, setStages] = useState<ResearchStage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeStage, setActiveStage] = useState<number>(1);

  const fetchStages = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/research/stages');
      if (!response.ok) {
        throw new Error('Failed to fetch research stages');
      }
      const data = await response.json();
      setStages(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchStages();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400">
        <p className="font-medium italic">Research Methodology visualization is currently disabled in settings.</p>
      </div>
    );
  }

  const currentStageData = stages.find(s => s.id === activeStage);

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-emerald-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">Q</div>
          <h3 className="text-lg font-bold text-slate-800">Research-Quest methodology</h3>
        </div>
        <span className="text-[10px] font-bold text-slate-400 uppercase bg-slate-50 px-2 py-1 rounded border">8-Stage Lifecycle</span>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {stages.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-in fade-in slide-in-from-top-2">
          {/* Stage Selection */}
          <div className="lg:col-span-1 flex lg:flex-col gap-2 overflow-x-auto pb-2 lg:pb-0">
            {stages.map((stage) => (
              <button
                key={stage.id}
                onClick={() => setActiveStage(stage.id)}
                className={`flex-none lg:flex-1 text-left px-3 py-2 rounded-lg border transition-all ${
                  activeStage === stage.id 
                    ? 'bg-emerald-600 text-white border-emerald-700 shadow-md transform scale-[1.02]' 
                    : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    activeStage === stage.id ? 'bg-white/20' : 'bg-slate-200'
                  }`}>
                    {stage.id}
                  </span>
                  <span className="text-xs font-bold truncate">{stage.name}</span>
                </div>
              </button>
            ))}
          </div>

          {/* Stage Details */}
          <div className="lg:col-span-3 bg-slate-50 rounded-xl border border-slate-200 p-5 space-y-6">
            {currentStageData && (
              <>
                <div>
                  <h4 className="text-xl font-bold text-slate-800">{currentStageData.name}</h4>
                  <p className="text-sm text-slate-500 mt-1">{currentStageData.description}</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <h5 className="text-[10px] font-bold text-emerald-600 uppercase tracking-widest border-b border-emerald-100 pb-1">Stage Objectives</h5>
                    <ul className="space-y-2">
                      {currentStageData.objectives.map((obj, i) => (
                        <li key={i} className="flex gap-2 items-start text-xs text-slate-700">
                          <span className="text-emerald-500 mt-0.5">●</span>
                          {obj}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="space-y-3">
                    <h5 className="text-[10px] font-bold text-blue-600 uppercase tracking-widest border-b border-blue-100 pb-1">Expected Outputs</h5>
                    <ul className="space-y-2">
                      {currentStageData.outputs.map((out, i) => (
                        <li key={i} className="flex gap-2 items-start text-xs text-slate-700">
                          <span className="text-blue-500 mt-0.5">■</span>
                          {out}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="p-4 bg-white rounded-lg border border-slate-200 shadow-sm">
                  <h5 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Quality Assurance Checks</h5>
                  <div className="space-y-2">
                    {currentStageData.quality_checks.map((check, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <div className="w-4 h-4 rounded border border-emerald-200 bg-emerald-50 flex items-center justify-center text-[10px] text-emerald-600">✓</div>
                        <span className="text-xs text-slate-600 font-medium">{check}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
