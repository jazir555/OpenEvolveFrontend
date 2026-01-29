import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';

export const RoadmapAgentViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.autogptEnabled || state.features.autogenEnabled || 
                    state.features.metagptEnabled || state.features.aiScientistEnabled ||
                    state.features.uncertainpyEnabled || state.features.riskAnalyzerEnabled ||
                    state.features.llm4iasEnabled || state.features.claraverseEnabled;

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">Roadmap Agent (Category 9) visualizations are currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gradient-to-tr from-purple-600 to-pink-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">9</div>
          <h3 className="text-lg font-bold text-slate-800 tracking-tight">Roadmap Agent Control Plane</h3>
        </div>
        <span className="text-[10px] font-bold text-slate-400 uppercase bg-slate-50 px-2 py-1 rounded border">Category 9 Synergy</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {state.features.autogptEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">AutoGPT Swarm</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Autonomous Task Loops</p>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-xs font-mono text-slate-600 tracking-tighter">SWARM_ACTIVE // LOOP_ID: {Math.random().toString(36).substring(7)}</span>
            </div>
          </div>
        )}

        {state.features.autogenEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">Microsoft AutoGen</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Conversation Dynamics</p>
            <div className="flex gap-1">
              <div className="w-6 h-6 rounded bg-blue-100 flex items-center justify-center text-[10px]">A1</div>
              <div className="w-6 h-6 rounded bg-indigo-100 flex items-center justify-center text-[10px]">A2</div>
              <div className="w-6 h-6 rounded bg-violet-100 flex items-center justify-center text-[10px]">A3</div>
            </div>
          </div>
        )}

        {state.features.metagptEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">MetaGPT Firm</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Software Company Simulation</p>
            <div className="space-y-1">
              <div className="flex justify-between text-[8px] font-bold text-slate-400">
                <span>PROJECT_ALPHA</span>
                <span>85%</span>
              </div>
              <div className="w-full h-1 bg-slate-200 rounded-full">
                <div className="h-full bg-indigo-500 w-[85%]" />
              </div>
            </div>
          </div>
        )}

        {state.features.aiScientistEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">AI Scientist</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Automated Hypothesizing</p>
            <div className="p-2 bg-white rounded border border-dashed flex flex-col gap-1">
              <span className="text-[9px] font-bold text-indigo-600 uppercase">New Hypothesis</span>
              <span className="text-[10px] text-slate-600 italic leading-tight">Neural-topological alignment improves zero-shot transfer.</span>
            </div>
          </div>
        )}

        {state.features.uncertainpyEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">Uncertainty Analysis</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Sensitivity Propagation</p>
            <div className="flex items-center justify-between">
              <div className="flex gap-0.5 items-end h-6">
                {[0.4, 0.7, 0.3, 0.9, 0.5].map((h, i) => (
                  <div key={i} className="w-2 bg-indigo-400 rounded-t-sm" style={{ height: `${h * 100}%` }} />
                ))}
              </div>
              <span className="text-[10px] font-mono text-slate-600">Var: 0.02</span>
            </div>
          </div>
        )}

        {state.features.riskAnalyzerEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">LLM Risk Analyzer</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Vulnerability Detection</p>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-sm" />
              <span className="text-[10px] font-bold text-emerald-600 uppercase">Status: Secure</span>
            </div>
          </div>
        )}

        {state.features.llm4iasEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">SOP Optimization</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">Procedure Enhancement</p>
            <span className="text-[10px] font-bold text-indigo-600 tracking-tighter uppercase">+15.2% Efficiency Gain</span>
          </div>
        )}

        {state.features.claraverseEnabled && (
          <div className="p-4 border rounded-xl bg-slate-50 relative group">
            <h4 className="text-sm font-bold text-slate-800 mb-1">Integration Assessment</h4>
            <p className="text-[10px] text-slate-500 mb-3 uppercase font-bold">ClaraVerse Compatibility</p>
            <div className="flex justify-between items-center text-[10px] font-bold text-slate-400">
              <span>92% Verified</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
