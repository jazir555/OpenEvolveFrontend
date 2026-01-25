import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GenericResult } from '../types/plugin-types';

export const QAComplianceViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const [qaData, setQaData] = useState<GenericResult | null>(null);
  const [redData, setRedData] = useState<GenericResult | null>(null);
  const [blueData, setBlueData] = useState<GenericResult | null>(null);
  const [reseData, setReseData] = useState<GenericResult | null>(null);

  const fetchData = async () => {
    if (state.features.qaSuiteEnabled) {
      const res = await fetch('/api/openevolve/qa/summary');
      setQaData(await res.json());
    }
    if (state.features.redTeamEnabled) {
      const res = await fetch('/api/openevolve/security/red-team');
      setRedData(await res.json());
    }
    if (state.features.blueTeamEnabled) {
      const res = await fetch('/api/openevolve/security/blue-team');
      setBlueData(await res.json());
    }
    if (state.features.reseEnabled) {
      const res = await fetch('/api/openevolve/rese/reliability');
      setReseData(await res.json());
    }
  };

  useEffect(() => {
    fetchData();
  }, [state.features.qaSuiteEnabled, state.features.redTeamEnabled, state.features.blueTeamEnabled, state.features.reseEnabled]);

  return (
    <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
      {state.features.qaSuiteEnabled && qaData && (
        <div className="p-4 border rounded-xl bg-white shadow-sm flex flex-col gap-4">
          <h3 className="text-lg font-bold text-slate-800">QA Suite Framework</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-emerald-50 rounded-lg">
              <p className="text-[10px] font-bold text-emerald-600 uppercase">Passed</p>
              <p className="text-2xl font-bold text-emerald-700">{qaData.passed}</p>
            </div>
            <div className="p-3 bg-rose-50 rounded-lg">
              <p className="text-[10px] font-bold text-rose-600 uppercase">Failed</p>
              <p className="text-2xl font-bold text-rose-700">{qaData.failed}</p>
            </div>
          </div>
          <div className="text-center text-[10px] font-bold text-slate-400 uppercase">
            Coverage: {((qaData.coverage || 0) * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {state.features.reseEnabled && reseData && (
        <div className="p-4 border rounded-xl bg-slate-900 text-white shadow-lg border-indigo-500/30">
          <h3 className="text-lg font-bold text-indigo-400 mb-4 flex justify-between items-center">
            RESE Reliability
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-ping" />
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-end">
              <span className="text-[10px] font-bold text-slate-500 uppercase">Score</span>
              <span className="text-3xl font-mono text-emerald-400 font-bold">{reseData.reliability_score?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase">
              <span>Error Rate: {reseData.error_rate}</span>
              <span>Uptime: {((reseData.uptime || 0) * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      )}

      {state.features.redTeamEnabled && redData && (
        <div className="p-4 border rounded-xl bg-rose-950 text-rose-100 border-rose-900 shadow-xl">
          <h3 className="text-lg font-bold mb-2">Red Team Attacks</h3>
          <div className="flex items-end gap-2">
            <span className="text-4xl font-black">{redData.attacks}</span>
            <span className="text-xs font-bold text-rose-400 uppercase pb-1">Attempts Logged</span>
          </div>
          <p className="text-[10px] font-bold text-rose-500 mt-2 uppercase tracking-widest">Severity: {redData.severity}</p>
        </div>
      )}

      {state.features.blueTeamEnabled && blueData && (
        <div className="p-4 border rounded-xl bg-indigo-950 text-indigo-100 border-indigo-900 shadow-xl">
          <h3 className="text-lg font-bold mb-2">Blue Team Shields</h3>
          <div className="flex flex-wrap gap-2 mt-3">
            {(blueData.defenses || []).map((d: string) => (
              <span key={d} className="px-2 py-1 bg-indigo-800 rounded text-[10px] font-bold uppercase tracking-tighter">
                {d}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
