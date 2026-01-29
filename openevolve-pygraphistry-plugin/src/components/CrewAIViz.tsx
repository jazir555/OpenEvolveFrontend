import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { CrewAIResult } from '../types/plugin-types';

export const CrewAIViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.crewaiEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CrewAIResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchCrewData = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/crewai/monitor');
      if (!response.ok) {
        throw new Error('Failed to fetch CrewAI data');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled) {
      fetchCrewData();
      const interval = setInterval(fetchCrewData, 8000); // 8s refresh
      return () => clearInterval(interval);
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-orange-100 rounded-lg bg-orange-50/30 text-orange-400">
        <p className="font-medium italic">Multi-AI Agent Orchestration is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-orange-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">C</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">CrewAI Team Orchestrator</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-orange-600 uppercase tracking-widest">{result?.crew_name || 'Autonomous Crew'}</span>
          <div className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-top-2">
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Active Agents</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {result.agents.map((agent, i) => (
                <div key={i} className="p-3 border rounded-xl bg-slate-50 flex items-center gap-3 group hover:border-orange-200 transition-colors">
                  <div className="w-10 h-10 rounded-full bg-white border border-slate-200 flex items-center justify-center text-xl grayscale group-hover:grayscale-0 transition-all">
                    🤖
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-bold text-slate-800 truncate">{agent.role}</p>
                    <p className="text-[10px] text-slate-500 italic truncate">{agent.goal}</p>
                  </div>
                  <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${
                    agent.status === 'working' ? 'bg-orange-50 text-orange-600 border-orange-100 animate-pulse' : 'bg-slate-100 text-slate-500 border-slate-200'
                  }`}>
                    {agent.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center px-1">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Orchestration Progress</h4>
              <span className="text-xs font-bold text-slate-700">{(result.progress * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden shadow-inner">
              <div className="bg-orange-500 h-full transition-all duration-1000" style={{ width: `${result.progress * 100}%` }} />
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest px-1">Task Pipeline</h4>
            <div className="space-y-2">
              {result.tasks.map((task, i) => (
                <div key={i} className="p-3 border rounded-lg bg-white flex items-center justify-between shadow-sm">
                  <div className="flex flex-col gap-0.5">
                    <p className="text-xs font-medium text-slate-700">{task.description}</p>
                    <p className="text-[10px] text-slate-400 font-bold uppercase tracking-tighter">Assigned: {task.agent}</p>
                  </div>
                  <span className={`text-[9px] font-bold px-2 py-0.5 rounded border uppercase ${
                    task.status === 'done' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 
                    task.status === 'in_progress' ? 'bg-orange-50 text-orange-600 border-orange-100' : 
                    'bg-slate-50 text-slate-400 border-slate-100'
                  }`}>
                    {task.status.replace('_', ' ')}
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
