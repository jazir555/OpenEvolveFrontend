import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { ACEResult } from '../types/plugin-types';

export const ACEViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.aceEnabled;

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ACEResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/openevolve/ace/analytics');
      if (!response.ok) {
        throw new Error('Failed to fetch ACE analytics');
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
      fetchAnalytics();
    }
  }, [isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">ACE Analytics visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-bold text-slate-800 uppercase tracking-tight">Agentic Context Engine (ACE) Analytics</h3>
        <button 
          onClick={fetchAnalytics}
          disabled={loading}
          className="text-xs bg-slate-100 hover:bg-slate-200 px-2 py-1 rounded transition-colors font-bold text-slate-600"
        >
          {loading ? 'Refreshing...' : 'Refresh Data'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-2">
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-indigo-600 uppercase tracking-widest border-b pb-1">Top Performing Teams</h4>
            <div className="space-y-3">
              {result.top_teams.map((team, i) => (
                <div key={i} className="p-3 bg-slate-50 rounded-lg border flex justify-between items-center">
                  <div>
                    <p className="text-sm font-bold text-slate-700">{team.team_name}</p>
                    <p className="text-[10px] text-slate-400 font-medium">Quality Score: {team.avg_quality_score.toFixed(2)}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-mono font-bold text-indigo-600">{(team.success_rate * 100).toFixed(0)}%</p>
                    <p className="text-[8px] uppercase font-bold text-slate-400">Success Rate</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="text-xs font-bold text-teal-600 uppercase tracking-widest border-b pb-1">Gauntlet Effectiveness</h4>
            <div className="space-y-3">
              {result.top_gauntlets.map((gauntlet, i) => (
                <div key={i} className="p-3 bg-slate-50 rounded-lg border">
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm font-bold text-slate-700">{gauntlet.gauntlet_name}</p>
                    <span className="text-[10px] bg-teal-100 text-teal-700 px-1.5 py-0.5 rounded font-bold uppercase">Active</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] font-bold text-slate-500">
                        <span>Detection Rate</span>
                        <span>{(gauntlet.detection_rate * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-1 rounded-full overflow-hidden">
                        <div className="bg-teal-500 h-full" style={{ width: `${gauntlet.detection_rate * 100}%` }} />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] font-bold text-slate-500">
                        <span>Precision</span>
                        <span>{(gauntlet.precision * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-1 rounded-full overflow-hidden">
                        <div className="bg-indigo-500 h-full" style={{ width: `${gauntlet.precision * 100}%` }} />
                      </div>
                    </div>
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
