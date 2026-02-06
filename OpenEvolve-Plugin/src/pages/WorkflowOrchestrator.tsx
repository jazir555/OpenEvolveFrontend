/**
 * OpenEvolve Workflow Orchestrator Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based orchestrator UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Settings, 
  Plus, 
  Trash2, 
  Edit3,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  Target,
  Users,
  Zap,
  GitBranch
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  parameters: Record<string, any>;
}

interface WorkflowInstance {
  id: string;
  name: string;
  templateId: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  startTime?: Date;
  endTime?: Date;
  parameters: Record<string, any>;
}

interface WorkflowParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'select';
  defaultValue: any;
  options?: string[];
  description?: string;
}

const WorkflowOrchestrator: React.FC = () => {
  const [activeTab, setActiveTab] = useState('create');
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([
    {
      id: 'template-1',
      name: 'Evolution Workflow',
      description: 'Standard evolutionary algorithm for optimization problems',
      category: 'optimization',
      parameters: {
        populationSize: 100,
        generations: 50,
        mutationRate: 0.1,
        crossoverRate: 0.8
      }
    },
    {
      id: 'template-2',
      name: 'Adversarial Testing',
      description: 'Red team/blue team adversarial validation',
      category: 'validation',
      parameters: {
        attackStrength: 0.5,
        numAttacks: 10,
        defenseStrategy: 'robust'
      }
    },
    {
      id: 'template-3',
      name: 'Decomposition Analysis',
      description: 'Problem decomposition and solution synthesis',
      category: 'analysis',
      parameters: {
        maxDepth: 3,
        granularity: 'medium',
        parallelProcessing: true
      }
    }
  ]);
  
  const [instances, setInstances] = useState<WorkflowInstance[]>([
    {
      id: 'instance-1',
      name: 'Optimization Run #1',
      templateId: 'template-1',
      status: 'running',
      progress: 45,
      startTime: new Date(Date.now() - 3600000),
      parameters: {
        populationSize: 100,
        generations: 50,
        mutationRate: 0.1,
        crossoverRate: 0.8
      }
    },
    {
      id: 'instance-2',
      name: 'Adversarial Test #1',
      templateId: 'template-2',
      status: 'completed',
      progress: 100,
      startTime: new Date(Date.now() - 7200000),
      endTime: new Date(Date.now() - 3600000),
      parameters: {
        attackStrength: 0.5,
        numAttacks: 10,
        defenseStrategy: 'robust'
      }
    }
  ]);
  
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [newInstanceName, setNewInstanceName] = useState('');
  const [newInstanceParams, setNewInstanceParams] = useState<Record<string, any>>({};
  const [editingInstanceId, setEditingInstanceId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load template parameters when template is selected
  useEffect(() => {
    if (selectedTemplate) {
      const template = templates.find(t => t.id === selectedTemplate);
      if (template) {
        setNewInstanceParams(template.parameters);
      }
    }
  }, [selectedTemplate, templates]);

  const handleCreateInstance = () => {
    if (!selectedTemplate || !newInstanceName.trim()) {
      alert('Please select a template and enter a name for the workflow instance');
      return;
    }

    const newInstanceId = `instance-${Date.now()}`;
    const newWorkflow: WorkflowInstance = {
      id: newInstanceId,
      name: newInstanceName,
      templateId: selectedTemplate,
      status: 'idle',
      progress: 0,
      parameters: { ...newInstanceParams }
    };

    setInstances([...instances, newWorkflow]);
    setNewInstanceName('');
    setSelectedTemplate('');
    setNewInstanceParams({});
  };

  const handleStartWorkflow = (instanceId: string) => {
    setInstances(instances.map(instance => 
      instance.id === instanceId 
        ? { ...instance, status: 'running', startTime: new Date() } 
        : instance
    ));
  };

  const handlePauseWorkflow = (instanceId: string) => {
    setInstances(instances.map(instance => 
      instance.id === instanceId 
        ? { ...instance, status: 'paused' } 
        : instance
    ));
  };

  const handleStopWorkflow = (instanceId: string) => {
    setInstances(instances.map(instance => 
      instance.id === instanceId 
        ? { ...instance, status: 'completed', endTime: new Date() } 
        : instance
    ));
  };

  const handleDeleteInstance = (instanceId: string) => {
    setInstances(instances.filter(instance => instance.id !== instanceId));
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

  const getActionButtons = (instance: WorkflowInstance) => {
    if (instance.status === 'running') {
      return (
        <div className="flex space-x-2">
          <BubbleButton
            variant="outline"
            size="sm"
            onClick={() => handlePauseWorkflow(instance.id)}
            className="flex items-center"
          >
            <Pause className="w-4 h-4 mr-1" />
            Pause
          </BubbleButton>
          <BubbleButton
            variant="destructive"
            size="sm"
            onClick={() => handleStopWorkflow(instance.id)}
            className="flex items-center"
          >
            <XCircle className="w-4 h-4 mr-1" />
            Stop
          </BubbleButton>
        </div>
      );
    } else if (instance.status === 'paused') {
      return (
        <div className="flex space-x-2">
          <BubbleButton
            variant="default"
            size="sm"
            onClick={() => handleStartWorkflow(instance.id)}
            className="flex items-center"
          >
            <Play className="w-4 h-4 mr-1" />
            Resume
          </BubbleButton>
          <BubbleButton
            variant="destructive"
            size="sm"
            onClick={() => handleStopWorkflow(instance.id)}
            className="flex items-center"
          >
            <XCircle className="w-4 h-4 mr-1" />
            Stop
          </BubbleButton>
        </div>
      );
    } else if (instance.status === 'idle') {
      return (
        <BubbleButton
          variant="default"
          size="sm"
          onClick={() => handleStartWorkflow(instance.id)}
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
          onClick={() => handleStartWorkflow(instance.id)}
          className="flex items-center"
        >
          <RotateCcw className="w-4 h-4 mr-1" />
          Restart
        </BubbleButton>
      );
    }
  };

  return (
    <div className="workflow-orchestrator p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <GitBranch className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Workflow Orchestrator
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
          Create, manage, and monitor OpenEvolve workflow instances
        </p>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="create" label="Create Workflow">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Template Selection */}
            <div className="lg:col-span-1">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2" />
                Workflow Templates
              </h2>
              <div className="space-y-4">
                {templates.map(template => (
                  <div 
                    key={template.id}
                    className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                      selectedTemplate === template.id 
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                        : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                    }`}
                    onClick={() => setSelectedTemplate(template.id)}
                  >
                    <h3 className="font-medium text-gray-900 dark:text-white">{template.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{template.description}</p>
                    <span className="inline-block mt-2 px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded">
                      {template.category}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Configuration */}
            <div className="lg:col-span-2">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Configuration
              </h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Workflow Instance Name
                  </label>
                  <BubbleInput
                    type="text"
                    value={newInstanceName}
                    onChange={(e) => setNewInstanceName(e.target.value)}
                    placeholder="Enter a name for this workflow instance"
                    className="w-full"
                  />
                </div>

                {selectedTemplate && (
                  <div>
                    <h3 className="text-md font-medium text-gray-900 dark:text-white mb-3">
                      Parameters for "{templates.find(t => t.id === selectedTemplate)?.name}"
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(newInstanceParams).map(([key, value]) => (
                        <div key={key}>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            {key}
                          </label>
                          {typeof value === 'boolean' ? (
                            <BubbleSelect
                              value={value.toString()}
                              onChange={(e) => setNewInstanceParams({
                                ...newInstanceParams,
                                [key]: e.target.value === 'true'
                              })}
                              className="w-full"
                            >
                              <option value="true">True</option>
                              <option value="false">False</option>
                            </BubbleSelect>
                          ) : typeof value === 'number' ? (
                            <BubbleInput
                              type="number"
                              value={value}
                              onChange={(e) => setNewInstanceParams({
                                ...newInstanceParams,
                                [key]: Number(e.target.value)
                              })}
                              className="w-full"
                            />
                          ) : (
                            <BubbleInput
                              type="text"
                              value={value}
                              onChange={(e) => setNewInstanceParams({
                                ...newInstanceParams,
                                [key]: e.target.value
                              })}
                              className="w-full"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="pt-4">
                  <BubbleButton
                    variant="default"
                    onClick={handleCreateInstance}
                    disabled={!selectedTemplate || !newInstanceName.trim()}
                    className="flex items-center"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Create Workflow Instance
                  </BubbleButton>
                </div>
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="monitoring" label="Monitoring">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Workflow</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Template</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Progress</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {instances.map(instance => {
                  const template = templates.find(t => t.id === instance.templateId);
                  return (
                    <tr key={instance.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900 dark:text-white">{instance.name}</div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">ID: {instance.id}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900 dark:text-white">{template?.name}</div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">{template?.category}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          instance.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                          instance.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                          instance.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                          instance.status === 'paused' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                        }`}>
                          {getStatusText(instance.status)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${instance.progress}%` }}
                            ></div>
                          </div>
                          <span className="text-sm text-gray-900 dark:text-white">{Math.round(instance.progress)}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {instance.startTime 
                          ? instance.endTime 
                            ? `${Math.round((instance.endTime.getTime() - instance.startTime.getTime()) / 60000)} min` 
                            : `${Math.round((Date.now() - instance.startTime.getTime()) / 60000)} min (running)`
                          : '-'
                        }
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <div className="flex items-center space-x-2">
                          {getActionButtons(instance)}
                          <button 
                            className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                            onClick={() => handleDeleteInstance(instance.id)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="history" label="History">
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <BubbleCard className="p-5">
                <div className="flex items-center">
                  <Zap className="w-8 h-8 text-blue-500 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Workflows</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">{instances.length}</p>
                  </div>
                </div>
              </BubbleCard>

              <BubbleCard className="p-5">
                <div className="flex items-center">
                  <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Completed</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {instances.filter(i => i.status === 'completed').length}
                    </p>
                  </div>
                </div>
              </BubbleCard>

              <BubbleCard className="p-5">
                <div className="flex items-center">
                  <XCircle className="w-8 h-8 text-red-500 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Failed</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {instances.filter(i => i.status === 'failed').length}
                    </p>
                  </div>
                </div>
              </BubbleCard>

              <BubbleCard className="p-5">
                <div className="flex items-center">
                  <BarChart3 className="w-8 h-8 text-purple-500 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {instances.length > 0 
                        ? `${Math.round((instances.filter(i => i.status === 'completed').length / instances.length) * 100)}%` 
                        : '0%'}
                    </p>
                  </div>
                </div>
              </BubbleCard>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Workflow History</h3>
              <div className="space-y-4">
                {instances.slice(0, 5).map(instance => {
                  const template = templates.find(t => t.id === instance.templateId);
                  return (
                    <div 
                      key={instance.id} 
                      className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white">{instance.name}</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">{template?.name}</p>
                        </div>
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(instance.status)}`}></div>
                      </div>
                      <div className="mt-3">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-600 dark:text-gray-400">Progress</span>
                          <span className="font-medium text-gray-900 dark:text-white">{Math.round(instance.progress)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${instance.progress}%` }}
                          ></div>
                        </div>
                      </div>
                      <div className="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                        <span>Started: {instance.startTime?.toLocaleString()}</span>
                        <span>Duration: {instance.endTime 
                          ? `${Math.round((instance.endTime.getTime() - instance.startTime!.getTime()) / 60000)} min` 
                          : `${Math.round((Date.now() - instance.startTime!.getTime()) / 60000)} min (running)`
                        }</span>
                      </div>
                    </div>
                  );
                })}
                
                {instances.length === 0 && (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    No workflow history available
                  </div>
                )}
              </div>
            </div>
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
      </div>
    </div>
  );
};

export default WorkflowOrchestrator;
