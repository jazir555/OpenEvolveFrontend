/**
 * OpenEvolve Main Application Page for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based main application UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Zap, 
  Shield, 
  Wrench, 
  Activity, 
  BarChart3, 
  Settings, 
  Play, 
  Pause, 
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Users,
  Target,
  TrendingUp,
  GitBranch,
  BookOpen,
  Database,
  Layout,
  Workflow,
  Search,
  Plus,
  Edit3,
  Trash2,
  Download,
  Upload,
  Filter,
  Eye,
  EyeOff
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';
import MainLayout from '../components/MainLayout';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  parameters: Record<string, any>;
}

interface RecentWorkflow {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  startTime: Date;
  endTime?: Date;
  duration?: number;
}

interface SystemStatus {
  apiStatus: 'online' | 'offline' | 'degraded';
  databaseStatus: 'connected' | 'disconnected' | 'slow';
  cacheStatus: 'healthy' | 'degraded' | 'error';
  workers: number;
  activeWorkflows: number;
  queuedTasks: number;
}

const MainApplication: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([
    {
      id: 'template-1',
      name: 'Evolution Pipeline',
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
  
  const [recentWorkflows, setRecentWorkflows] = useState<RecentWorkflow[]>([
    {
      id: 'wf-001',
      name: 'Optimization Run #1',
      status: 'completed',
      progress: 100,
      startTime: new Date(Date.now() - 7200000),
      endTime: new Date(Date.now() - 3600000),
      duration: 3600
    },
    {
      id: 'wf-002',
      name: 'Adversarial Test #1',
      status: 'running',
      progress: 65,
      startTime: new Date(Date.now() - 3600000),
      duration: 3600
    },
    {
      id: 'wf-003',
      name: 'Decomposition Analysis #1',
      status: 'failed',
      progress: 30,
      startTime: new Date(Date.now() - 1800000),
      endTime: new Date(Date.now() - 1200000),
      duration: 600
    }
  ]);
  
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    apiStatus: 'online',
    databaseStatus: 'connected',
    cacheStatus: 'healthy',
    workers: 4,
    activeWorkflows: 2,
    queuedTasks: 5
  });
  
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [newWorkflowName, setNewWorkflowName] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Simulate loading data
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const handleCreateWorkflow = () => {
    if (!newWorkflowName.trim() || !selectedTemplate) {
      alert('Please enter a name and select a template');
      return;
    }

    const newWorkflow: RecentWorkflow = {
      id: `wf-${Date.now()}`,
      name: newWorkflowName,
      status: 'idle',
      progress: 0,
      startTime: new Date()
    };

    setRecentWorkflows([newWorkflow, ...recentWorkflows]);
    setNewWorkflowName('');
    setSelectedTemplate('');
  };

  const handleStartWorkflow = (workflowId: string) => {
    setRecentWorkflows(recentWorkflows.map(wf => 
      wf.id === workflowId 
        ? { ...wf, status: 'running', startTime: new Date() } 
        : wf
    ));
  };

  const handleStopWorkflow = (workflowId: string) => {
    setRecentWorkflows(recentWorkflows.map(wf => 
      wf.id === workflowId 
        ? { ...wf, status: 'completed', endTime: new Date() } 
        : wf
    ));
  };

  const handleDeleteWorkflow = (workflowId: string) => {
    setRecentWorkflows(recentWorkflows.filter(wf => wf.id !== workflowId));
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
      case 'idle': return 'Idle';
      default: return status;
    }
  };

  const getApiStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      case 'degraded': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getDatabaseStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-500';
      case 'disconnected': return 'bg-red-500';
      case 'slow': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading OpenEvolve application...</span>
      </div>
    );
  }

  return (
    <MainLayout>
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Content Improver
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
            <BubbleButton
              variant="default"
              size="sm"
              className="flex items-center"
            >
              <Play className="w-4 h-4 mr-2" />
              New Workflow
            </BubbleButton>
          </div>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          AI-powered content improvement using evolutionary algorithms and adversarial testing
        </p>
      </div>

      {/* System Status Bar */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-6">
        <div className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className={`w-3 h-3 rounded-full mr-2 ${getApiStatusColor(systemStatus.apiStatus)}`}></div>
          <span className="text-sm font-medium text-gray-900 dark:text-white">API</span>
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400 capitalize">{systemStatus.apiStatus}</span>
        </div>
        
        <div className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className={`w-3 h-3 rounded-full mr-2 ${getDatabaseStatusColor(systemStatus.databaseStatus)}`}></div>
          <span className="text-sm font-medium text-gray-900 dark:text-white">Database</span>
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400 capitalize">{systemStatus.databaseStatus}</span>
        </div>
        
        <div className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className={`w-3 h-3 rounded-full mr-2 ${getDatabaseStatusColor(systemStatus.cacheStatus)}`}></div>
          <span className="text-sm font-medium text-gray-900 dark:text-white">Cache</span>
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400 capitalize">{systemStatus.cacheStatus}</span>
        </div>
        
        <div className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <Users className="w-4 h-4 text-gray-600 dark:text-gray-400 mr-2" />
          <span className="text-sm font-medium text-gray-900 dark:text-white">Workers</span>
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400">{systemStatus.workers}</span>
        </div>
        
        <div className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <Activity className="w-4 h-4 text-gray-600 dark:text-gray-400 mr-2" />
          <span className="text-sm font-medium text-gray-900 dark:text-white">Active</span>
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400">{systemStatus.activeWorkflows}</span>
        </div>
        
        <div className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <Clock className="w-4 h-4 text-gray-600 dark:text-gray-400 mr-2" />
          <span className="text-sm font-medium text-gray-900 dark:text-white">Queued</span>
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400">{systemStatus.queuedTasks}</span>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div 
          className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 hover:shadow-md transition-shadow cursor-pointer"
          onClick={() => setActiveTab('workflows')}
        >
          <div className="flex items-center">
            <Workflow className="w-8 h-8 text-blue-500 mr-3" />
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">New Workflow</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">Create a new workflow</p>
            </div>
          </div>
        </div>

        <div 
          className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 hover:shadow-md transition-shadow cursor-pointer"
          onClick={() => setActiveTab('evolution')}
        >
          <div className="flex items-center">
            <Zap className="w-8 h-8 text-yellow-500 mr-3" />
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">Evolution</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">Run evolutionary algorithms</p>
            </div>
          </div>
        </div>

        <div 
          className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 hover:shadow-md transition-shadow cursor-pointer"
          onClick={() => setActiveTab('adversarial')}
        >
          <div className="flex items-center">
            <Shield className="w-8 h-8 text-red-500 mr-3" />
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">Adversarial</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">Test with adversarial methods</p>
            </div>
          </div>
        </div>

        <div 
          className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 hover:shadow-md transition-shadow cursor-pointer"
          onClick={() => setActiveTab('analytics')}
        >
          <div className="flex items-center">
            <BarChart3 className="w-8 h-8 text-purple-500 mr-3" />
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">Analytics</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">View performance metrics</p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="dashboard" label="Dashboard">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Recent Workflows */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Clock className="w-5 h-5 mr-2" />
                Recent Workflows
              </h2>
              <div className="space-y-4">
                {recentWorkflows.map(workflow => (
                  <div 
                    key={workflow.id} 
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">{workflow.name}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">ID: {workflow.id}</p>
                      </div>
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(workflow.status)}`}></div>
                    </div>
                    <div className="mt-3">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">Progress</span>
                        <span className="font-medium text-gray-900 dark:text-white">{Math.round(workflow.progress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full" 
                          style={{ width: `${workflow.progress}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                      <span>Started: {workflow.startTime.toLocaleTimeString()}</span>
                      <span>
                        {workflow.duration 
                          ? `${Math.round(workflow.duration / 60)} min` 
                          : workflow.endTime 
                            ? `${Math.round((workflow.endTime.getTime() - workflow.startTime.getTime()) / 60000)} min` 
                            : `${Math.round((Date.now() - workflow.startTime.getTime()) / 60000)} min (running)`
                        }
                      </span>
                    </div>
                    <div className="mt-3 flex space-x-2">
                      {workflow.status === 'idle' || workflow.status === 'paused' ? (
                        <BubbleButton
                          variant="default"
                          size="sm"
                          onClick={() => handleStartWorkflow(workflow.id)}
                          className="flex items-center"
                        >
                          <Play className="w-4 h-4 mr-1" />
                          Start
                        </BubbleButton>
                      ) : workflow.status === 'running' ? (
                        <BubbleButton
                          variant="destructive"
                          size="sm"
                          onClick={() => handleStopWorkflow(workflow.id)}
                          className="flex items-center"
                        >
                          <XCircle className="w-4 h-4 mr-1" />
                          Stop
                        </BubbleButton>
                      ) : (
                        <BubbleButton
                          variant="outline"
                          size="sm"
                          onClick={() => handleStartWorkflow(workflow.id)}
                          className="flex items-center"
                        >
                          <RotateCcw className="w-4 h-4 mr-1" />
                          Restart
                        </BubbleButton>
                      )}
                      <button 
                        className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                        onClick={() => handleDeleteWorkflow(workflow.id)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
                
                {recentWorkflows.length === 0 && (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    No recent workflows
                  </div>
                )}
              </div>
            </div>

            {/* Workflow Templates */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Layout className="w-5 h-5 mr-2" />
                Workflow Templates
              </h2>
              <div className="space-y-4">
                {templates.map(template => (
                  <div 
                    key={template.id} 
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">{template.name}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{template.description}</p>
                      </div>
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                        {template.category}
                      </span>
                    </div>
                    <div className="mt-3">
                      <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">Parameters:</h4>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        {Object.entries(template.parameters).map(([key, value]) => (
                          <div key={key} className="flex justify-between py-1">
                            <span className="capitalize">{key.replace(/([A-Z])/g, ' $1')}:</span>
                            <span className="font-medium">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="mt-4">
                      <BubbleButton
                        variant="outline"
                        size="sm"
                        className="w-full"
                        onClick={() => {
                          setSelectedTemplate(template.id);
                          setNewWorkflowName(template.name + ' - ' + new Date().toISOString().split('T')[0]);
                        }}
                      >
                        Use Template
                      </BubbleButton>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="workflows" label="Workflows">
          <div className="mb-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
              <div className="flex-1">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-5 w-5 text-gray-400" />
                  </div>
                  <BubbleInput
                    type="text"
                    placeholder="Search workflows..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 w-full"
                  />
                </div>
              </div>
              <div className="flex space-x-2">
                <BubbleSelect
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-40"
                >
                  <option value="all">All Categories</option>
                  <option value="optimization">Optimization</option>
                  <option value="validation">Validation</option>
                  <option value="analysis">Analysis</option>
                </BubbleSelect>
                <BubbleButton
                  variant="default"
                  className="flex items-center"
                  onClick={() => setActiveTab('create')}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  New
                </BubbleButton>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recentWorkflows.map(workflow => (
                <div 
                  key={workflow.id} 
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">{workflow.name}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">ID: {workflow.id}</p>
                    </div>
                    <div className={`w-3 h-3 rounded-full ${getStatusColor(workflow.status)}`}></div>
                  </div>
                  <div className="mt-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">Progress</span>
                      <span className="font-medium text-gray-900 dark:text-white">{Math.round(workflow.progress)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${workflow.progress}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>Status: {getStatusText(workflow.status)}</span>
                    <span>
                      {workflow.duration 
                        ? `${Math.round(workflow.duration / 60)} min` 
                        : workflow.endTime 
                          ? `${Math.round((workflow.endTime.getTime() - workflow.startTime.getTime()) / 60000)} min` 
                          : `${Math.round((Date.now() - workflow.startTime.getTime()) / 60000)} min (running)`
                      }
                    </span>
                  </div>
                  <div className="mt-4 flex space-x-2">
                    <BubbleButton
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      View
                    </BubbleButton>
                    <BubbleButton
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      Logs
                    </BubbleButton>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="create" label="Create Workflow">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Template Selection */}
            <div className="lg:col-span-1">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Layout className="w-5 h-5 mr-2" />
                Select Template
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
                    Workflow Name
                  </label>
                  <BubbleInput
                    type="text"
                    value={newWorkflowName}
                    onChange={(e) => setNewWorkflowName(e.target.value)}
                    placeholder="Enter a name for this workflow"
                    className="w-full"
                  />
                </div>

                {selectedTemplate && (
                  <div>
                    <h3 className="text-md font-medium text-gray-900 dark:text-white mb-3">
                      Parameters for "{templates.find(t => t.id === selectedTemplate)?.name}"
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(templates.find(t => t.id === selectedTemplate)?.parameters || {}).map(([key, value]) => (
                        <div key={key}>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            {key.replace(/([A-Z])/g, ' $1')}
                          </label>
                          {typeof value === 'boolean' ? (
                            <BubbleSelect
                              value={value.toString()}
                              onChange={(e) => {
                                // Update template parameters
                              }}
                              className="w-full"
                            >
                              <option value="true">True</option>
                              <option value="false">False</option>
                            </BubbleSelect>
                          ) : typeof value === 'number' ? (
                            <BubbleInput
                              type="number"
                              value={value}
                              onChange={(e) => {
                                // Update template parameters
                              }}
                              className="w-full"
                            />
                          ) : (
                            <BubbleInput
                              type="text"
                              value={value}
                              onChange={(e) => {
                                // Update template parameters
                              }}
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
                    onClick={handleCreateWorkflow}
                    disabled={!selectedTemplate || !newWorkflowName.trim()}
                    className="flex items-center"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Create Workflow
                  </BubbleButton>
                </div>
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
        <BubbleButton
          variant="default"
          className="flex items-center"
        >
          <Play className="w-4 h-4 mr-2" />
          New Workflow
        </BubbleButton>
      </div>
    </MainLayout>
  );
};

export default MainApplication;
