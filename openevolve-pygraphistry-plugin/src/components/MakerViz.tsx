import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GenericResult } from '../types/plugin-types';

export const MakerViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.makerEnabled || state.features.mdapEnabled;
  const [makerData, setMakerData] = useState<GenericResult | null>(null);
  const [mdapData, setMdapData] = useState<GenericResult | null>(null);
  const [mctsData, setMctsData] = useState<GenericResult | null>(null);
  const [hybridData, setHybridData] = useState<GenericResult | null>(null);
  const [karateclubData, setKarateclubData] = useState<GenericResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    if (!isEnabled) {
      return;
    }

    setError(null);
    if (state.features.makerEnabled) {
      try {
        const res = await fetch('/api/openevolve/maker/voting');
        if (!res.ok) {
          throw new Error('Failed to fetch MAKER results');
        }
        setMakerData(await res.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch MAKER results');
      }
    }
    if (state.features.mdapEnabled) {
      try {
        const res = await fetch('/api/openevolve/mdap/processing');
        if (!res.ok) {
          throw new Error('Failed to fetch MDAP results');
        }
        setMdapData(await res.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch MDAP results');
      }
    }
    if (state.features.mctsEnabled) {
      try {
        const res = await fetch('/api/openevolve/mcts/search');
        if (!res.ok) {
          throw new Error('Failed to fetch MCTS results');
        }
        setMctsData(await res.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch MCTS results');
      }
    }
    if (state.features.hybridMCTSEnabled) {
      try {
        const res = await fetch('/api/openevolve/mcts/hybrid');
        if (!res.ok) {
          throw new Error('Failed to fetch hybrid MCTS results');
        }
        setHybridData(await res.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch hybrid MCTS results');
      }
    }
    if (state.features.karateclubEnabled) {
      try {
        const res = await fetch('/api/openevolve/graph/ml');
        if (!res.ok) {
          throw new Error('Failed to fetch graph ML results');
        }
        setKarateclubData(await res.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch graph ML results');
      }
    }
  };

  useEffect(() => {
    fetchData();
  }, [state.features.makerEnabled, state.features.mdapEnabled, state.features.mctsEnabled, 
      state.features.hybridMCTSEnabled, state.features.karateclubEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400">
        <p className="font-medium italic">MDAP/MAKER visualization is currently disabled in settings.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 p-4">
      {error && (
        <div className="p-3 bg-rose-50 text-rose-700 border border-rose-200 rounded text-sm">
          {error}
        </div>
      )}
      {state.features.karateclubEnabled && karateclubData && (
        <div className="p-4 border rounded-xl bg-white shadow-sm border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-2">KarateClub Graph ML</h3>
          <div className="flex justify-between items-center">
            <span className="text-xs font-bold text-indigo-600 uppercase tracking-tighter">{karateclubData.algorithm}</span>
            <span className="text-[10px] font-bold text-slate-400">{karateclubData.clusters} Communities Detected</span>
          </div>
        </div>
      )}

      {state.features.hybridMCTSEnabled && hybridData && (
        <div className="p-4 border rounded-xl bg-gradient-to-r from-indigo-900 to-slate-900 text-white shadow-lg">
          <h3 className="text-lg font-bold text-indigo-300 mb-2">Hybrid MCTS (Synergy)</h3>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] font-bold text-slate-500 uppercase">Evolution Count</p>
              <p className="text-2xl font-black">{hybridData.evolution_count}</p>
            </div>
            <div className="text-right">
              <p className="text-[10px] font-bold text-slate-500 uppercase">Hybrid Score</p>
              <p className="text-2xl font-black text-emerald-400">{hybridData.hybrid_score?.toFixed(4)}</p>
            </div>
          </div>
        </div>
      )}

      {state.features.mdapEnabled && mdapData && (
        <div className="p-4 border rounded-xl bg-white shadow-sm">
          <h3 className="text-lg font-bold text-slate-800 mb-2">MDAP Multi-Dim Processing</h3>
          <div className="flex flex-wrap gap-2">
            {(mdapData.dimensions || []).map((dim: string, i: number) => (
              <div key={dim} className="px-3 py-1 bg-indigo-50 border border-indigo-100 rounded-full flex items-center gap-2">
                <span className="text-xs font-bold text-indigo-700 uppercase">{dim}</span>
                <span className="text-xs font-mono text-indigo-400">{((mdapData.scores?.[i] || 0) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {state.features.mctsEnabled && mctsData && (
        <div className="p-4 border rounded-xl bg-slate-900 text-white shadow-lg">
          <h3 className="text-lg font-bold text-indigo-400 mb-2">MCTS Tree Search</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-[10px] font-bold text-slate-500 uppercase">Nodes Explored</p>
              <p className="text-2xl font-mono font-bold">{mctsData.nodes_explored?.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-[10px] font-bold text-slate-500 uppercase">Best Reward</p>
              <p className="text-2xl font-mono font-bold text-emerald-400">{mctsData.best_reward?.toFixed(4)}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
