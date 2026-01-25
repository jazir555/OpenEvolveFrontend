import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GenericResult } from '../types/plugin-types';

export const ScientificDiscoveryViz: React.FC = () => {
  const state = pygraphistryPlugin.getState();
  const [materialData, setMaterialData] = useState<GenericResult | null>(null);
  const [gnomeData, setGnomeData] = useState<GenericResult | null>(null);
  const [physicsData, setPhysicsData] = useState<GenericResult | null>(null);
  const [uqData, setUqData] = useState<GenericResult | null>(null);
  const [pylabData, setPylabData] = useState<GenericResult | null>(null);
  const [pinnsData, setPinnsData] = useState<GenericResult | null>(null);
  const [neuralkgData, setNeuralkgData] = useState<GenericResult | null>(null);

  const fetchData = async () => {
    if (state.features.materialKGEnabled) {
      const res = await fetch('/api/openevolve/material/kg');
      setMaterialData(await res.json());
    }
    if (state.features.gnomeEnabled) {
      const res = await fetch('/api/openevolve/discovery/gnome');
      setGnomeData(await res.json());
    }
    if (state.features.physicsNemoEnabled) {
      const res = await fetch('/api/openevolve/physics/nemo');
      setPhysicsData(await res.json());
    }
    if (state.features.uqEnabled) {
      const res = await fetch('/api/openevolve/uq/analyze');
      setUqData(await res.json());
    }
    if (state.features.pylabrobotEnabled) {
      const res = await fetch('/api/openevolve/robotics/pylabrobot');
      setPylabData(await res.json());
    }
    if (state.features.pinnsEnabled) {
      const res = await fetch('/api/openevolve/physics/pinns');
      setPinnsData(await res.json());
    }
    if (state.features.neuralKGEnabled) {
      const res = await fetch('/api/openevolve/graph/neuralkg');
      setNeuralkgData(await res.json());
    }
  };

  useEffect(() => {
    fetchData();
  }, [state.features.materialKGEnabled, state.features.gnomeEnabled, state.features.physicsNemoEnabled, 
      state.features.uqEnabled, state.features.pylabrobotEnabled, state.features.pinnsEnabled, state.features.neuralKGEnabled]);

  return (
    <div className="p-4 grid grid-cols-1 gap-6">
      {state.features.gnomeEnabled && gnomeData && (
        <div className="p-6 border rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-700 text-white shadow-xl">
          <div className="flex justify-between items-start mb-6">
            <h3 className="text-xl font-black uppercase tracking-tighter">GNoME Materials Discovery</h3>
            <span className="text-[10px] font-bold bg-white/20 px-2 py-1 rounded border border-white/30 uppercase">Screening</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
            <div>
              <p className="text-[10px] font-bold text-indigo-200 uppercase mb-1">Candidates</p>
              <p className="text-3xl font-black">{gnomeData.candidate_materials?.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-[10px] font-bold text-indigo-200 uppercase mb-1">Validated</p>
              <p className="text-3xl font-black text-emerald-300">{gnomeData.valid_materials}</p>
            </div>
            <div className="hidden md:block">
              <p className="text-[10px] font-bold text-indigo-200 uppercase mb-1">Success Rate</p>
              <p className="text-3xl font-black text-white/80">{((gnomeData.valid_materials / (gnomeData.candidate_materials || 1)) * 100).toFixed(2)}%</p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {state.features.pylabrobotEnabled && pylabData && (
          <div className="p-4 border rounded-xl bg-slate-50 border-slate-200 shadow-sm">
            <h3 className="text-lg font-bold text-slate-800 mb-2">PyLabRobot Automation</h3>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] font-bold text-slate-400 uppercase">Status</p>
                <p className="text-sm font-bold text-emerald-600 uppercase">{pylabData.status}</p>
              </div>
              <div className="text-right">
                <p className="text-[10px] font-bold text-slate-400 uppercase">Plates</p>
                <p className="text-xl font-black text-slate-800">{pylabData.plates}</p>
              </div>
            </div>
          </div>
        )}

        {state.features.pinnsEnabled && pinnsData && (
          <div className="p-4 border rounded-xl bg-white shadow-sm border-indigo-100">
            <h3 className="text-lg font-bold text-slate-800 mb-2">PINNs Physics ML</h3>
            <div className="space-y-1">
              <p className="text-[8px] font-bold text-slate-400 uppercase">PDE Residual</p>
              <p className="text-lg font-mono font-bold text-indigo-600">{pinnsData.pde_residual}</p>
            </div>
          </div>
        )}

        {state.features.neuralKGEnabled && neuralkgData && (
          <div className="p-4 border rounded-xl bg-slate-900 text-white shadow-lg">
            <h3 className="text-lg font-bold text-indigo-400 mb-2">NeuralKG Embedding</h3>
            <div className="flex justify-between items-center">
              <span className="text-[10px] font-bold text-slate-500 uppercase">{neuralkgData.algorithm}</span>
              <span className="text-xl font-black text-emerald-400">{neuralkgData.dim}d</span>
            </div>
          </div>
        )}
      </div>

      {state.features.uqEnabled && uqData && (
        <div className="p-4 border rounded-xl bg-white shadow-sm border-indigo-100">
          <h3 className="text-lg font-bold text-slate-800 mb-4">Uncertainty Quantification (UQ)</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-slate-50 rounded-lg">
              <p className="text-[8px] font-bold text-slate-400 uppercase">Variance</p>
              <p className="text-xl font-mono font-bold text-indigo-600">{uqData.statistics?.std?.toFixed(6)}</p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg">
              <p className="text-[8px] font-bold text-slate-400 uppercase">Confidence</p>
              <p className="text-xl font-mono font-bold text-emerald-600">0.99</p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {state.features.materialKGEnabled && materialData && (
          <div className="p-4 border rounded-xl bg-white shadow-sm border-slate-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Material Knowledge Graph</h3>
            <div className="flex items-center justify-between">
              <div className="text-center">
                <p className="text-[10px] font-bold text-slate-400 uppercase">Compounds</p>
                <p className="text-2xl font-bold text-slate-800">{materialData.compounds?.toLocaleString()}</p>
              </div>
              <div className="flex flex-wrap gap-1 justify-end max-w-[150px]">
                {(materialData.properties || []).map((p: string) => (
                  <span key={p} className="px-2 py-0.5 bg-slate-100 rounded text-[8px] font-bold text-slate-500 uppercase">{p}</span>
                ))}
              </div>
            </div>
          </div>
        )}

        {state.features.physicsNemoEnabled && physicsData && (
          <div className="p-4 border rounded-xl bg-slate-50 border-slate-200 shadow-sm">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Physics-NeMo Simulation</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-500 font-bold uppercase">Convergence</span>
                <span className="font-mono font-bold text-indigo-600">{((physicsData.convergence || 0) * 100).toFixed(2)}%</span>
              </div>
              <div className="w-full h-1 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-full bg-indigo-500" style={{ width: `${(physicsData.convergence || 0) * 100}%` }} />
              </div>
              <div className="flex justify-between items-center text-[10px]">
                <span className="text-slate-400 font-medium">Error Norm: {physicsData.error_norm}</span>
                <span className="text-slate-400 font-medium">Sims: {physicsData.simulations}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
