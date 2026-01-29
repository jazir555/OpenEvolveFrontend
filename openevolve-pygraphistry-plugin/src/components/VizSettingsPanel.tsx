import React, { useState } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { PyGraphistryPluginState } from '../types/plugin-types';

export const VizSettingsPanel: React.FC = () => {
  const [state, setState] = useState<PyGraphistryPluginState>(pygraphistryPlugin.getState());

  const toggleFeature = (key: keyof PyGraphistryPluginState['features']) => {
    const newFeatures = { ...state.features, [key]: !state.features[key] };
    pygraphistryPlugin.updateFeatures(newFeatures);
    setState(pygraphistryPlugin.getState());
  };

  const handleConfigChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    pygraphistryPlugin.updateConfig({ [name]: value });
    setState(pygraphistryPlugin.getState());
  };

  return (
    <div className="p-6 bg-white rounded-xl shadow-sm border border-slate-200 space-y-6">
      <div>
        <h3 className="text-lg font-bold text-slate-800">Visualization Settings</h3>
        <p className="text-sm text-slate-500 text-balance">
          Manage visualization components and their respective backend configurations.
        </p>
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Feature Toggles</h4>
        
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">PyGraphistry</p>
            <p className="text-xs text-slate-500">Interactive Knowledge Graphs</p>
          </div>
          <button 
            onClick={() => toggleFeature('pygraphistryEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.pygraphistryEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.pygraphistryEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Causal Discovery</p>
            <p className="text-xs text-slate-500">Identify Causal Mechanisms</p>
          </div>
          <button 
            onClick={() => toggleFeature('causalDiscoveryEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.causalDiscoveryEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.causalDiscoveryEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">NeuroMANCER</p>
            <p className="text-xs text-slate-500">Loss Landscape Visualization</p>
          </div>
          <button 
            onClick={() => toggleFeature('optimizationEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.optimizationEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.optimizationEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Uncertainty Quantification</p>
            <p className="text-xs text-slate-500">Sensitivity Analysis (uqtestfuns)</p>
          </div>
          <button 
            onClick={() => toggleFeature('uqEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.uqEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.uqEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Chemical Knowledge</p>
            <p className="text-xs text-slate-500">Molecular Explorer (GlobalChem)</p>
          </div>
          <button 
            onClick={() => toggleFeature('globalChemEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.globalChemEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.globalChemEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Scientific Experimentation</p>
            <p className="text-xs text-slate-500">Automated Protocols (Curie)</p>
          </div>
          <button 
            onClick={() => toggleFeature('curieEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.curieEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.curieEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Temporal Knowledge Graph</p>
            <p className="text-xs text-slate-500">Time-aware Facts (Graphiti)</p>
          </div>
          <button 
            onClick={() => toggleFeature('temporalGraphEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.temporalGraphEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.temporalGraphEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Knowledge Extraction</p>
            <p className="text-xs text-slate-500">Schema-guided NER/RE (OneKE)</p>
          </div>
          <button 
            onClick={() => toggleFeature('onekeEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.onekeEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.onekeEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Autoformalization</p>
            <p className="text-xs text-slate-500">Math to Lean4 Proofs (LeanAide)</p>
          </div>
          <button 
            onClick={() => toggleFeature('leanAideEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.leanAideEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.leanAideEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">SOP Generation</p>
            <p className="text-xs text-slate-500">Structured Operating Procedures</p>
          </div>
          <button 
            onClick={() => toggleFeature('sopEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.sopEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.sopEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Adversarial Validation</p>
            <p className="text-xs text-slate-500">Red/Blue Team Robustness</p>
          </div>
          <button 
            onClick={() => toggleFeature('adversarialEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.adversarialEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.adversarialEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Pattern Mining</p>
            <p className="text-xs text-slate-500">Frequent Itemset Discovery (PAMI)</p>
          </div>
          <button 
            onClick={() => toggleFeature('pamiEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.pamiEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.pamiEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Context Analytics</p>
            <p className="text-xs text-slate-500">Team & Gauntlet Performance (ACE)</p>
          </div>
          <button 
            onClick={() => toggleFeature('aceEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.aceEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.aceEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Recursive Meta-Agents</p>
            <p className="text-xs text-slate-500">Hierarchical Orchestration (ROMA)</p>
          </div>
          <button 
            onClick={() => toggleFeature('romaEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.romaEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.romaEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Multi-Agent Data Processing</p>
            <p className="text-xs text-slate-500">Blue/Red/Gold Workflow (DataPizza)</p>
          </div>
          <button 
            onClick={() => toggleFeature('datapizzaEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.datapizzaEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.datapizzaEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Active Reliability</p>
            <p className="text-xs text-slate-500">Deterministic Verification (ACE + Steer)</p>
          </div>
          <button 
            onClick={() => toggleFeature('steerEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.steerEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.steerEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Project Management</p>
            <p className="text-xs text-slate-500">Ticket Tracking (Hephaestus)</p>
          </div>
          <button 
            onClick={() => toggleFeature('hephaestusEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.hephaestusEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.hephaestusEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Autonomous Development</p>
            <p className="text-xs text-slate-500">Task Decomposition (Claudiomiro)</p>
          </div>
          <button 
            onClick={() => toggleFeature('claudiomiroEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.claudiomiroEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.claudiomiroEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Research Methodology</p>
            <p className="text-xs text-slate-500">8-Stage Lifecycle (Research-Quest)</p>
          </div>
          <button 
            onClick={() => toggleFeature('researchQuestEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.researchQuestEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.researchQuestEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Knowledge Graph Generation</p>
            <p className="text-xs text-slate-500">Text to KG (KG-GEN)</p>
          </div>
          <button 
            onClick={() => toggleFeature('kgEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.kgEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.kgEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Sovereign Workflow Monitoring</p>
            <p className="text-xs text-slate-500">Real-time SGD Metrics (Advanced Monitoring)</p>
          </div>
          <button 
            onClick={() => toggleFeature('sgdEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.sgdEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.sgdEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Global Performance Analytics</p>
            <p className="text-xs text-slate-500">Cross-Project Cost & Token Tracking</p>
          </div>
          <button 
            onClick={() => toggleFeature('globalAnalyticsEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.globalAnalyticsEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.globalAnalyticsEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Quality-Diversity (MAP-Elites)</p>
            <p className="text-xs text-slate-500">Feature Space Optimization</p>
          </div>
          <button 
            onClick={() => toggleFeature('mapElitesEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.mapElitesEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.mapElitesEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Mathematical Verification</p>
            <p className="text-xs text-slate-500">Algorithmic Correctness Analysis</p>
          </div>
          <button 
            onClick={() => toggleFeature('verificationEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.verificationEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.verificationEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Problem Analysis</p>
            <p className="text-xs text-slate-500">Semantic Problem Decomposition</p>
          </div>
          <button 
            onClick={() => toggleFeature('problemAnalysisEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.problemAnalysisEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.problemAnalysisEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Dependency Mapping</p>
            <p className="text-xs text-slate-500">Sub-problem DAG Visualization</p>
          </div>
          <button 
            onClick={() => toggleFeature('dependencyEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.dependencyEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.dependencyEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Knowledge Artifact Mapping</p>
            <p className="text-xs text-slate-500">Relationship Graph (Artifact Manager)</p>
          </div>
          <button 
            onClick={() => toggleFeature('artifactGraphEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.artifactGraphEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.artifactGraphEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Symbolic Logic Constraints</p>
            <p className="text-xs text-slate-500">Formal Logic & Lean4 Verification (SCE)</p>
          </div>
          <button 
            onClick={() => toggleFeature('sceEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.sceEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.sceEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Static Code Analysis</p>
            <p className="text-xs text-slate-500">Security & Quality Scanning (DeepStatic)</p>
          </div>
          <button 
            onClick={() => toggleFeature('staticAnalysisEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.staticAnalysisEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.staticAnalysisEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Logic-to-Loss Translation</p>
            <p className="text-xs text-slate-500">Differentiable Constraints (LLTL)</p>
          </div>
          <button 
            onClick={() => toggleFeature('lltlEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.lltlEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.lltlEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Multi-Agent Collaboration</p>
            <p className="text-xs text-slate-500">Real-time Session Sync & Conflict Resolution</p>
          </div>
          <button 
            onClick={() => toggleFeature('collaborationEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.collaborationEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.collaborationEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Workflow Execution Monitor</p>
            <p className="text-xs text-slate-500">Real-time Pipeline & Resource Tracking</p>
          </div>
          <button 
            onClick={() => toggleFeature('workflowMonitorEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.workflowMonitorEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.workflowMonitorEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Evolution Ancestry & Lineage</p>
            <p className="text-xs text-slate-500">Parent-Child Improvement Tracing</p>
          </div>
          <button 
            onClick={() => toggleFeature('lineageEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.lineageEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.lineageEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Gauntlet Effectiveness</p>
            <p className="text-xs text-slate-500">Validation Catch-Rate Analysis</p>
          </div>
          <button 
            onClick={() => toggleFeature('gauntletEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.gauntletEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.gauntletEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Knowledge Pattern Discovery</p>
            <p className="text-xs text-slate-500">ML-based Solution Clustering (Miner)</p>
          </div>
          <button 
            onClick={() => toggleFeature('patternMiningEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.patternMiningEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.patternMiningEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Dynamic Gauntlet Adaptation</p>
            <p className="text-xs text-slate-500">Active Validation Strictness Optimization</p>
          </div>
          <button 
            onClick={() => toggleFeature('adaptationEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.adaptationEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.adaptationEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">High-Performance Logic Audit</p>
            <p className="text-xs text-slate-500">O(n log n) Contradiction Detection (DITO)</p>
          </div>
          <button 
            onClick={() => toggleFeature('ditoEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.ditoEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.ditoEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Knowledge Retrieval (RAG)</p>
            <p className="text-xs text-slate-500">Augmented Context Recovery (Ragbits)</p>
          </div>
          <button 
            onClick={() => toggleFeature('ragEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.ragEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.ragEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Knowledge Extraction</p>
            <p className="text-xs text-slate-500">Structured Entity & Relation Discovery (DeepKE)</p>
          </div>
          <button 
            onClick={() => toggleFeature('deepkeEnabled')}
            className={`w-12 h-6 rounded-full transition-colors relative ${state.features.deepkeEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}
          >
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${state.features.deepkeEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        {/* DECISION & SEARCH ENGINES */}
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6">Decision & Search Engines</h4>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Multi-Agent Voting</p>
            <p className="text-xs text-slate-500">Consensus & Proposals (MAKER)</p>
          </div>
          <button onClick={() => toggleFeature('makerEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.makerEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.makerEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Multi-Dim Processing</p>
            <p className="text-xs text-slate-500">Dimensional Synthesis (MDAP)</p>
          </div>
          <button onClick={() => toggleFeature('mdapEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.mdapEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.mdapEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Monte Carlo Tree Search</p>
            <p className="text-xs text-slate-500">Decision Space Optimization (MCTS)</p>
          </div>
          <button onClick={() => toggleFeature('mctsEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.mctsEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.mctsEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Hybrid MCTS</p>
            <p className="text-xs text-slate-500">Evolution + Search Synergy</p>
          </div>
          <button onClick={() => toggleFeature('hybridMCTSEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.hybridMCTSEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.hybridMCTSEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        {/* QUALITY & RELIABILITY */}
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6">Quality & Reliability</h4>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Multi-Judge Evaluator</p>
            <p className="text-xs text-slate-500">Consensus Scoring (Evaluator Team)</p>
          </div>
          <button onClick={() => toggleFeature('evaluatorTeamEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.evaluatorTeamEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.evaluatorTeamEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Red Team Security</p>
            <p className="text-xs text-slate-500">Autonomous Vulnerability Probing</p>
          </div>
          <button onClick={() => toggleFeature('redTeamEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.redTeamEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.redTeamEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Blue Team Defense</p>
            <p className="text-xs text-slate-500">Security Hardening & Shielding</p>
          </div>
          <button onClick={() => toggleFeature('blueTeamEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.blueTeamEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.blueTeamEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">QA Suite</p>
            <p className="text-xs text-slate-500">Comprehensive Test Coverage</p>
          </div>
          <button onClick={() => toggleFeature('qaSuiteEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.qaSuiteEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.qaSuiteEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">RESE Reliability</p>
            <p className="text-xs text-slate-500">System-wide Fault Tolerance</p>
          </div>
          <button onClick={() => toggleFeature('reseEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.reseEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.reseEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        {/* SCIENTIFIC & DISCOVERY */}
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6">Scientific & Discovery</h4>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Material KG</p>
            <p className="text-xs text-slate-500">Materials Science Knowledge Graph</p>
          </div>
          <button onClick={() => toggleFeature('materialKGEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.materialKGEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.materialKGEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">GNoME Discovery</p>
            <p className="text-xs text-slate-500">AI-driven Material Exploration</p>
          </div>
          <button onClick={() => toggleFeature('gnomeEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.gnomeEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.gnomeEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Physics-NeMo</p>
            <p className="text-xs text-slate-500">High-fidelity Simulation</p>
          </div>
          <button onClick={() => toggleFeature('physicsNemoEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.physicsNemoEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.physicsNemoEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">PINNs Library</p>
            <p className="text-xs text-slate-500">Physics-Informed Neural Networks</p>
          </div>
          <button onClick={() => toggleFeature('pinnsEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.pinnsEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.pinnsEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">PyLabRobot</p>
            <p className="text-xs text-slate-500">Laboratory Robotics Automation</p>
          </div>
          <button onClick={() => toggleFeature('pylabrobotEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.pylabrobotEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.pylabrobotEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        {/* GRAPH ML & EMBEDDINGS */}
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6">Graph ML & Embeddings</h4>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">KarateClub</p>
            <p className="text-xs text-slate-500">Unsupervised Graph ML (DeepWalk/node2vec)</p>
          </div>
          <button onClick={() => toggleFeature('karateclubEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.karateclubEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.karateclubEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">NeuralKG</p>
            <p className="text-xs text-slate-500">Knowledge Graph Embedding Framework</p>
          </div>
          <button onClick={() => toggleFeature('neuralKGEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.neuralKGEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.neuralKGEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        {/* ROADMAP AGENTS */}
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6">Roadmap Agents (Category 9)</h4>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">AutoGPT Swarms</p>
            <p className="text-xs text-slate-500">Autonomous Task Loops</p>
          </div>
          <button onClick={() => toggleFeature('autogptEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.autogptEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.autogptEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Microsoft AutoGen</p>
            <p className="text-xs text-slate-500">Multi-agent Conversation Dynamics</p>
          </div>
          <button onClick={() => toggleFeature('autogenEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.autogenEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.autogenEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">MetaGPT Firm</p>
            <p className="text-xs text-slate-500">Software Company Simulation</p>
          </div>
          <button onClick={() => toggleFeature('metagptEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.metagptEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.metagptEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">AI Scientist</p>
            <p className="text-xs text-slate-500">Automated Scientific Hypothesizing</p>
          </div>
          <button onClick={() => toggleFeature('aiScientistEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.aiScientistEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.aiScientistEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>

        {/* ERROR ANALYSIS & GAP FILLING */}
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6">Error Analysis & Gap Filling</h4>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Uncertainty Analysis</p>
            <p className="text-xs text-slate-500">Propagation & Sensitivity (Uncertainpy)</p>
          </div>
          <button onClick={() => toggleFeature('uncertainpyEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.uncertainpyEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.uncertainpyEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">LLM Risk Analyzer</p>
            <p className="text-xs text-slate-500">Vulnerability & Bias Detection</p>
          </div>
          <button onClick={() => toggleFeature('riskAnalyzerEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.riskAnalyzerEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.riskAnalyzerEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">SOP Optimization</p>
            <p className="text-xs text-slate-500">Procedure Enhancement (LLM4IAS)</p>
          </div>
          <button onClick={() => toggleFeature('llm4iasEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.llm4iasEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.llm4iasEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
          <div>
            <p className="font-medium text-slate-700">Integration Assessment</p>
            <p className="text-xs text-slate-500">ClaraVerse Compatibility Auditing</p>
          </div>
          <button onClick={() => toggleFeature('claraverseEnabled')} className={`w-12 h-6 rounded-full relative ${state.features.claraverseEnabled ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${state.features.claraverseEnabled ? 'translate-x-6' : ''}`} />
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Configuration</h4>
        
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-600 block">Graphistry API Key</label>
          <input 
            type="password"
            name="apiKey"
            value={state.config.apiKey || ''}
            onChange={handleConfigChange}
            placeholder="sk_..."
            className="w-full p-2 border rounded-md text-sm font-mono focus:ring-2 focus:ring-indigo-500 outline-none"
          />
        </div>
      </div>
    </div>
  );
};
