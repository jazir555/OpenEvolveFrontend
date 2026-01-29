/**
 * OpenEvolve Evolution Component for BubbleLab
 * 
 * This component replaces the Streamlit-based evolution UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import {
  Zap,
  Target,
  TrendingUp,
  Settings,
  Play,
  Pause,
  RotateCcw,
  BarChart3,
  Users,
  GitBranch,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface EvolutionConfig {
  populationSize: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  selectionMethod: string;
  elitismCount: number;
  tournamentSize: number;
  temperature: number;
  modelId: string;
  mdapMakerEnabled: boolean;
  mdapMakerAutoSelect: boolean;
}

interface EvolutionRun {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  generation: number;
  bestFitness: number;
  avgFitness: number;
  startTime?: Date;
  endTime?: Date;
  config: EvolutionConfig;
}

const EvolutionPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('configure');
  const [config, setConfig] = useState<EvolutionConfig>({
    populationSize: 100,
    generations: 50,
    mutationRate: 0.1,
    crossoverRate: 0.8,
    selectionMethod: 'tournament',
    elitismCount: 2,
    tournamentSize: 3,
    temperature: 0.7,
    modelId: 'gpt-4',
    mdapMakerEnabled: false,
    mdapMakerAutoSelect: true
  });
  
  const [runs, setRuns] = useState<EvolutionRun[]>([
    {
      id: 'run-1',
      name: 'Optimization Run #1',
      status: 'completed',
      progress: 100,
      generation: 50,
      bestFitness: 0.95,
      avgFitness: 0.82,
      startTime: new Date(Date.now() - 7200000),
      endTime: new Date(Date.now() - 3600000),
      config: {
        populationSize: 100,
        generations: 50,
        mutationRate: 0.1,
        crossoverRate: 0.8,
        selectionMethod: 'tournament',
        elitismCount: 2,
        tournamentSize: 3,
        temperature: 0.7,
        modelId: 'gpt-4',
        mdapMakerEnabled: false,
        mdapMakerAutoSelect: true
      }
    },
    {
      id: 'run-2',
      name: 'Optimization Run #2',
      status: 'running',
      progress: 65,
      generation: 32,
      bestFitness: 0.88,
      avgFitness: 0.75,
      startTime: new Date(Date.now() - 3600000),
      config: {
        populationSize: 150,
        generations: 100,
        mutationRate: 0.15,
        crossoverRate: 0.7,
        selectionMethod: 'roulette',
        elitismCount: 3,
        tournamentSize: 5,
        temperature: 0.8,
        modelId: 'gpt-3.5-turbo',
        mdapMakerEnabled: true,
        mdapMakerAutoSelect: false
      }
    }
  ]);
  
  const [newRunName, setNewRunName] = useState('');
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleStartRun = () => {
    if (!newRunName.trim()) {
      alert('Please enter a name for the evolution run');
      return;
    }

    const newRunId = `run-${Date.now()}`;
    const newRun: EvolutionRun = {
      id: newRunId,
      name: newRunName,
      status: 'running',
      progress: 0,
      generation: 0,
      bestFitness: 0,
      avgFitness: 0,
      startTime: new Date(),
      config: { ...config }
    };

    setRuns([newRun, ...runs]);
    setNewRunName('');
  };

  const handlePauseRun = (runId: string) => {
    setRuns(runs.map(run => 
      run.id === runId 
        ? { ...run, status: 'paused' } 
        : run
    ));
  };

  const handleResumeRun = (runId: string) => {
    setRuns(runs.map(run => 
      run.id === runId 
        ? { ...run, status: 'running' } 
        : run
    ));
  };

  const handleStopRun = (runId: string) => {
    setRuns(runs.map(run => 
      run.id === runId 
        ? { ...run, status: 'completed', endTime: new Date() } 
        : run
    ));
  };

  const handleDeleteRun = (runId: string) => {
    setRuns(runs.filter(run => run.id !== runId));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      case 'paused': return 'bg-yellow-500';
      case 'idle': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return 'Running';
      case 'completed': return 'Completed';
      case 'failed': return 'Failed';
      case 'paused': return 'Paused';
      case 'idle': return 'Ready';
      default: return status;
    }
  };

  const getActionButtons = (run: EvolutionRun) => {
    if (run.status === 'running') {
      return (
        <div className="flex space-x-2">
          <BubbleButton
            variant="outline"
            size="sm"
            onClick={() => handlePauseRun(run.id)}
            className="flex items-center"
          >
            <Pause className="w-4 h-4 mr-1" />
            Pause
          </BubbleButton>
          <BubbleButton
            variant="destructive"
            size="sm"
            onClick={() => handleStopRun(run.id)}
            className="flex items-center"
          >
            <XCircle className="w-4 h-4 mr-1" />
            Stop
          </BubbleButton>
        </div>
      );
    } else if (run.status === 'paused') {
      return (
        <div className="flex space-x-2">
          <BubbleButton
            variant="default"
            size="sm"
            onClick={() => handleResumeRun(run.id)}
            className="flex items-center"
          >
            <Play className="w-4 h-4 mr-1" />
            Resume
          </BubbleButton>
          <BubbleButton
            variant="destructive"
            size="sm"
            onClick={() => handleStopRun(run.id)}
            className="flex items-center"
          >
            <XCircle className="w-4 h-4 mr-1" />
            Stop
          </BubbleButton>
        </div>
      );
    } else if (run.status === 'idle') {
      return (
        <BubbleButton
          variant="default"
          size="sm"
          onClick={() => handleResumeRun(run.id)}
          className="flex items-center"
        >
          <Play className="w-4 h-4 mr-1" />
          Start
        </BubbleButton>
      );
    } else {
      return (
        <BubbleButton
          variant="outline"
          size="sm"
          onClick={() => handleResumeRun(run.id)}
          className="flex items-center"
        >
          <RotateCcw className="w-4 h-4 mr-1" />
          Restart
        </BubbleButton>
      );
    }
  };

  return (
    <div className="evolution-page p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Zap className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Evolution Engine
            </h1>
          </div>
          <div className="flex items-center space-x-3">
            <BubbleButton
              variant="outline"
              size="sm"
              className="flex items-center"
            >
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </BubbleButton>
          </div>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Configure and run evolutionary algorithms for optimization and improvement
        </p>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="configure" label="Configure">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Configuration Panel */}
            <div className="lg:col-span-2">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Evolution Configuration
              </h2>
              
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Population Size
                    </label>
                    <BubbleInput
                      type="number"
                      value={config.populationSize}
                      onChange={(e) => setConfig({...config, populationSize: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Generations
                    </label>
                    <BubbleInput
                      type="number"
                      value={config.generations}
                      onChange={(e) => setConfig({...config, generations: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Mutation Rate
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={config.mutationRate}
                      onChange={(e) => setConfig({...config, mutationRate: parseFloat(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Crossover Rate
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={config.crossoverRate}
                      onChange={(e) => setConfig({...config, crossoverRate: parseFloat(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Selection Method
                    </label>
                    <BubbleSelect
                      value={config.selectionMethod}
                      onChange={(e) => setConfig({...config, selectionMethod: e.target.value})}
                      className="w-full"
                    >
                      <option value="tournament">Tournament</option>
                      <option value="roulette">Roulette</option>
                      <option value="rank">Rank</option>
                      <option value="uniform">Uniform</option>
                    </BubbleSelect>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Tournament Size
                    </label>
                    <BubbleInput
                      type="number"
                      value={config.tournamentSize}
                      onChange={(e) => setConfig({...config, tournamentSize: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Elitism Count
                    </label>
                    <BubbleInput
                      type="number"
                      value={config.elitismCount}
                      onChange={(e) => setConfig({...config, elitismCount: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Temperature
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={config.temperature}
                      onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Model ID
                    </label>
                    <BubbleInput
                      type="text"
                      value={config.modelId}
                      onChange={(e) => setConfig({...config, modelId: e.target.value})}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* MDAP/MAKER Integration */}
                <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                  <h3 className="text-md font-medium text-gray-900 dark:text-white mb-3">
                    MDAP/MAKER Integration
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="mdapMakerEnabled"
                        checked={config.mdapMakerEnabled}
                        onChange={(e) => setConfig({...config, mdapMakerEnabled: e.target.checked})}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      <label htmlFor="mdapMakerEnabled" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Enable MDAP/MAKER
                      </label>
                    </div>

                    {config.mdapMakerEnabled && (
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="mdapMakerAutoSelect"
                          checked={config.mdapMakerAutoSelect}
                          onChange={(e) => setConfig({...config, mdapMakerAutoSelect: e.target.checked})}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                        />
                        <label htmlFor="mdapMakerAutoSelect" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                          Auto-Select for Critical Tasks
                        </label>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Run Panel */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Play className="w-5 h-5 mr-2" />
                Start New Run
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Run Name
                  </label>
                  <BubbleInput
                    type="text"
                    value={newRunName}
                    onChange={(e) => setNewRunName(e.target.value)}
                    placeholder="Enter a name for this run"
                    className="w-full"
                  />
                </div>

                <div className="pt-4">
                  <BubbleButton
                    variant="default"
                    onClick={handleStartRun}
                    disabled={!newRunName.trim()}
                    className="w-full flex items-center justify-center"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Start Evolution Run
                  </BubbleButton>
                </div>

                <div className="mt-6">
                  <h3 className="text-md font-medium text-gray-900 dark:text-white mb-3">
                    Current Configuration
                  </h3>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 text-sm">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Population:</span>
                        <span className="font-medium text-gray-900 dark:text-white">{config.populationSize}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Generations:</span>
                        <span className="font-medium text-gray-900 dark:text-white">{config.generations}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Mutation Rate:</span>
                        <span className="font-medium text-gray-900 dark:text-white">{config.mutationRate}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Crossover Rate:</span>
                        <span className="font-medium text-gray-900 dark:text-white">{config.crossoverRate}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Selection:</span>
                        <span className="font-medium text-gray-900 dark:text-white">{config.selectionMethod}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Model:</span>
                        <span className="font-medium text-gray-900 dark:text-white">{config.modelId}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="runs" label="Active Runs">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Run</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Progress</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Generation</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Best Fitness</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {runs.map(run => (
                  <tr key={run.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{run.name}</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">ID: {run.id}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        run.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                        run.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        run.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                        run.status === 'paused' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                        'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                      }`}>
                        {getStatusText(run.status)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${run.progress}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900 dark:text-white">{Math.round(run.progress)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {run.generation} / {run.config.generations}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {run.bestFitness.toFixed(4)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {run.startTime 
                        ? run.endTime 
                          ? `${Math.round((run.endTime.getTime() - run.startTime.getTime()) / 60000)} min` 
                          : `${Math.round((Date.now() - run.startTime.getTime()) / 60000)} min (running)`
                        : '-'
                      }
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex items-center space-x-2">
                        {getActionButtons(run)}
                        <button 
                          className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                          onClick={() => handleDeleteRun(run.id)}
                        >
                          <XCircle className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="analytics" label="Analytics">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <BubbleCard className="p-5">
              <div className="flex items-center">
                <Zap className="w-8 h-8 text-blue-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Runs</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.filter(r => r.status === 'running' || r.status === 'paused').length}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <Target className="w-8 h-8 text-green-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Completed Runs</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.filter(r => r.status === 'completed').length}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <TrendingUp className="w-8 h-8 text-purple-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg. Best Fitness</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.length > 0 
                      ? (runs.reduce((sum, r) => sum + r.bestFitness, 0) / runs.length).toFixed(4) 
                      : '0.0000'}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <BarChart3 className="w-8 h-8 text-yellow-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.length > 0 
                      ? `${Math.round((runs.filter(r => r.status === 'completed').length / runs.length) * 100)}%` 
                      : '0%'}
                  </p>
                </div>
              </div>
            </BubbleCard>
          </div>

          {/* Fitness Over Time Chart */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2" />
              Fitness Progression
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart
                data={runs.length > 0 ? (() => {
                  // Generate mock fitness data for the completed run
                  const completedRun = runs.find(r => r.status === 'completed') || runs[0];
                  const generations = completedRun?.generation || 50;
                  const data = [];
                  let bestFitness = 0.5;
                  let avgFitness = 0.3;

                  for (let i = 0; i <= generations; i += Math.max(1, Math.floor(generations / 20))) {
                    bestFitness = Math.min(0.98, bestFitness + Math.random() * 0.05);
                    avgFitness = Math.min(bestFitness - 0.1, avgFitness + Math.random() * 0.04);
                    data.push({
                      generation: i,
                      bestFitness: Number(bestFitness.toFixed(4)),
                      avgFitness: Number(avgFitness.toFixed(4))
                    });
                  }

                  return data;
                })() : [{ generation: 0, bestFitness: 0, avgFitness: 0 }]}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" className="dark:stroke-gray-700" />
                <XAxis
                  dataKey="generation"
                  stroke="#9CA3AF"
                  label={{ value: 'Generation', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                />
                <YAxis
                  stroke="#9CA3AF"
                  label={{ value: 'Fitness', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgb(31 41 55)',
                    border: '1px solid rgb(75 85 99)',
                    borderRadius: '0.5rem',
                    color: '#F9FAFB'
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="bestFitness"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Best Fitness"
                  dot={{ fill: '#10B981', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="avgFitness"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  name="Average Fitness"
                  dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </BubbleTab>
      </BubbleTabs>

      {/* Action Buttons */}
      <div className="mt-8 flex justify-end space-x-3">
        <BubbleButton
          variant="outline"
          className="flex items-center"
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Refresh
        </BubbleButton>
        <BubbleButton
          variant="default"
          className="flex items-center"
        >
          <Play className="w-4 h-4 mr-2" />
          New Run
        </BubbleButton>
      </div>
    </div>
  );
};

export default EvolutionPage;