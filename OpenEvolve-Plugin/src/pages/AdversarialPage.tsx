/**
 * OpenEvolve Adversarial Testing Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based adversarial UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import {
  Shield,
  Target,
  Zap,
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
  AlertTriangle,
  Eye,
  EyeOff
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface AdversarialConfig {
  enabled: boolean;
  attackStrategy: string;
  numExamples: number;
  strength: number;
  stepSize: number;
  numSteps: number;
  defenseStrategy: string;
  robustnessThreshold: number;
  modelId: string;
  mdapMakerEnabled: boolean;
  mdapMakerAutoSelect: boolean;
}

interface AdversarialRun {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  attackSuccessRate: number;
  defenseSuccessRate: number;
  startTime?: Date;
  endTime?: Date;
  config: AdversarialConfig;
}

const AdversarialPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('configure');
  const [config, setConfig] = useState<AdversarialConfig>({
    enabled: true,
    attackStrategy: 'pgd',
    numExamples: 10,
    strength: 0.3,
    stepSize: 0.01,
    numSteps: 40,
    defenseStrategy: 'robust',
    robustnessThreshold: 0.7,
    modelId: 'gpt-4',
    mdapMakerEnabled: false,
    mdapMakerAutoSelect: true
  });
  
  const [runs, setRuns] = useState<AdversarialRun[]>([
    {
      id: 'adv-run-1',
      name: 'Security Test #1',
      status: 'completed',
      progress: 100,
      attackSuccessRate: 0.15,
      defenseSuccessRate: 0.85,
      startTime: new Date(Date.now() - 7200000),
      endTime: new Date(Date.now() - 3600000),
      config: {
        enabled: true,
        attackStrategy: 'fgsm',
        numExamples: 20,
        strength: 0.2,
        stepSize: 0.01,
        numSteps: 20,
        defenseStrategy: 'robust',
        robustnessThreshold: 0.75,
        modelId: 'gpt-4',
        mdapMakerEnabled: false,
        mdapMakerAutoSelect: true
      }
    },
    {
      id: 'adv-run-2',
      name: 'Robustness Eval #1',
      status: 'running',
      progress: 75,
      attackSuccessRate: 0.12,
      defenseSuccessRate: 0.88,
      startTime: new Date(Date.now() - 3600000),
      config: {
        enabled: true,
        attackStrategy: 'pgd',
        numExamples: 15,
        strength: 0.25,
        stepSize: 0.02,
        numSteps: 30,
        defenseStrategy: 'certified',
        robustnessThreshold: 0.8,
        modelId: 'gpt-3.5-turbo',
        mdapMakerEnabled: true,
        mdapMakerAutoSelect: false
      }
    }
  ]);
  
  const [newRunName, setNewRunName] = useState('');
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const handleStartRun = () => {
    if (!newRunName.trim()) {
      alert('Please enter a name for the adversarial run');
      return;
    }

    const newRunId = `adv-run-${Date.now()}`;
    const newRun: AdversarialRun = {
      id: newRunId,
      name: newRunName,
      status: 'running',
      progress: 0,
      attackSuccessRate: 0,
      defenseSuccessRate: 0,
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

  const getActionButtons = (run: AdversarialRun) => {
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
    <div className="adversarial-page p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Shield className="w-8 h-8 text-red-600 dark:text-red-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Adversarial Testing
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
          Test and improve model robustness against adversarial attacks
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
                Adversarial Configuration
              </h2>
              
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Attack Strategy
                    </label>
                    <BubbleSelect
                      value={config.attackStrategy}
                      onChange={(e) => setConfig({...config, attackStrategy: e.target.value})}
                      className="w-full"
                    >
                      <option value="fgsm">FGSM (Fast Gradient Sign)</option>
                      <option value="pgd">PGD (Projected Gradient Descent)</option>
                      <option value="cw">Carlini-Wagner</option>
                      <option value="bim">BIM (Basic Iterative Method)</option>
                      <option value="deepfool">DeepFool</option>
                    </BubbleSelect>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Defense Strategy
                    </label>
                    <BubbleSelect
                      value={config.defenseStrategy}
                      onChange={(e) => setConfig({...config, defenseStrategy: e.target.value})}
                      className="w-full"
                    >
                      <option value="robust">Robust Training</option>
                      <option value="certified">Certified Defenses</option>
                      <option value="detection">Adversarial Detection</option>
                      <option value="randomization">Randomization</option>
                      <option value="gradient_masking">Gradient Masking</option>
                    </BubbleSelect>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Number of Examples
                    </label>
                    <BubbleInput
                      type="number"
                      value={config.numExamples}
                      onChange={(e) => setConfig({...config, numExamples: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Attack Strength
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={config.strength}
                      onChange={(e) => setConfig({...config, strength: parseFloat(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Step Size
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.001"
                      min="0"
                      max="1"
                      value={config.stepSize}
                      onChange={(e) => setConfig({...config, stepSize: parseFloat(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Number of Steps
                    </label>
                    <BubbleInput
                      type="number"
                      value={config.numSteps}
                      onChange={(e) => setConfig({...config, numSteps: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Robustness Threshold
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={config.robustnessThreshold}
                      onChange={(e) => setConfig({...config, robustnessThreshold: parseFloat(e.target.value) || 0})}
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
                        id="advMdapMakerEnabled"
                        checked={config.mdapMakerEnabled}
                        onChange={(e) => setConfig({...config, mdapMakerEnabled: e.target.checked})}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      <label htmlFor="advMdapMakerEnabled" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Enable MDAP/MAKER
                      </label>
                    </div>

                    {config.mdapMakerEnabled && (
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="advMdapMakerAutoSelect"
                          checked={config.mdapMakerAutoSelect}
                          onChange={(e) => setConfig({...config, mdapMakerAutoSelect: e.target.checked})}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                        />
                        <label htmlFor="advMdapMakerAutoSelect" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
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
                Start New Test
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Test Name
                  </label>
                  <BubbleInput
                    type="text"
                    value={newRunName}
                    onChange={(e) => setNewRunName(e.target.value)}
                    placeholder="Enter a name for this test"
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
                    Start Adversarial Test
                  </BubbleButton>
                </div>

                <div className="mt-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-md font-medium text-gray-900 dark:text-white">
                      Current Configuration
                    </h3>
                    <button 
                      onClick={() => setShowConfig(!showConfig)}
                      className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                    >
                      {showConfig ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  
                  {showConfig && (
                    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 text-sm">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Attack:</span>
                          <span className="font-medium text-gray-900 dark:text-white capitalize">{config.attackStrategy}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Defense:</span>
                          <span className="font-medium text-gray-900 dark:text-white capitalize">{config.defenseStrategy}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Examples:</span>
                          <span className="font-medium text-gray-900 dark:text-white">{config.numExamples}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Strength:</span>
                          <span className="font-medium text-gray-900 dark:text-white">{config.strength}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Steps:</span>
                          <span className="font-medium text-gray-900 dark:text-white">{config.numSteps}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Model:</span>
                          <span className="font-medium text-gray-900 dark:text-white">{config.modelId}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="runs" label="Active Tests">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Test</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Progress</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Attack Success</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Defense Success</th>
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
                      {(run.attackSuccessRate * 100).toFixed(2)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(run.defenseSuccessRate * 100).toFixed(2)}%
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
                <Shield className="w-8 h-8 text-red-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Tests</p>
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
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Completed Tests</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.filter(r => r.status === 'completed').length}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <Zap className="w-8 h-8 text-blue-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg. Defense Rate</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.length > 0 
                      ? `${(runs.reduce((sum, r) => sum + r.defenseSuccessRate, 0) / runs.length * 100).toFixed(2)}%` 
                      : '0.00%'}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <BarChart3 className="w-8 h-8 text-yellow-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg. Attack Rate</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {runs.length > 0 
                      ? `${(runs.reduce((sum, r) => sum + r.attackSuccessRate, 0) / runs.length * 100).toFixed(2)}%` 
                      : '0.00%'}
                  </p>
                </div>
              </div>
            </BubbleCard>
          </div>

          {/* Robustness Chart */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <Shield className="w-5 h-5 mr-2" />
              Robustness Analysis
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart
                data={runs.length > 0 ? runs.map(run => ({
                  name: run.name.substring(0, 15),
                  attackSuccess: Number((run.attackSuccessRate * 100).toFixed(2)),
                  defenseSuccess: Number((run.defenseSuccessRate * 100).toFixed(2))
                })) : [{ name: 'No Data', attackSuccess: 0, defenseSuccess: 0 }]}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" className="dark:stroke-gray-700" />
                <XAxis
                  dataKey="name"
                  stroke="#9CA3AF"
                  tick={{ fill: '#9CA3AF' }}
                />
                <YAxis
                  stroke="#9CA3AF"
                  label={{ value: 'Success Rate (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                  tick={{ fill: '#9CA3AF' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgb(31 41 55)',
                    border: '1px solid rgb(75 85 99)',
                    borderRadius: '0.5rem',
                    color: '#F9FAFB'
                  }}
                  formatter={(value: number) => [`${value}%`, '']}
                />
                <Legend />
                <Bar
                  dataKey="attackSuccess"
                  fill="#EF4444"
                  name="Attack Success Rate"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="defenseSuccess"
                  fill="#10B981"
                  name="Defense Success Rate"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
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
          New Test
        </BubbleButton>
      </div>
    </div>
  );
};

export default AdversarialPage;
