/**
 * OpenEvolve Workflow Builder Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based workflow builder UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import {
  GitBranch,
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
  Shield,
  BookOpen,
  Database,
  Layout,
  Workflow
} from 'lucide-react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  NodeTypes,
  EdgeTypes,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  MarkerType
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Custom node component
const CustomNode = ({ data }: { data: any }) => {
  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'start': return <GitBranch className="w-4 h-4 text-green-500" />;
      case 'end': return <GitBranch className="w-4 h-4 text-red-500" />;
      case 'evolution': return <Zap className="w-4 h-4 text-blue-500" />;
      case 'adversarial': return <Shield className="w-4 h-4 text-red-500" />;
      case 'decomposition': return <Target className="w-4 h-4 text-purple-500" />;
      case 'knowledge': return <BookOpen className="w-4 h-4 text-indigo-500" />;
      case 'leanaide': return <Database className="w-4 h-4 text-cyan-500" />;
      case 'crewai': return <Users className="w-4 h-4 text-orange-500" />;
      default: return <Layout className="w-4 h-4 text-gray-500" />;
    }
  };

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'start': return 'bg-green-50 border-green-300 dark:bg-green-900/20 dark:border-green-700';
      case 'end': return 'bg-red-50 border-red-300 dark:bg-red-900/20 dark:border-red-700';
      case 'evolution': return 'bg-blue-50 border-blue-300 dark:bg-blue-900/20 dark:border-blue-700';
      case 'adversarial': return 'bg-red-50 border-red-300 dark:bg-red-900/20 dark:border-red-700';
      case 'decomposition': return 'bg-purple-50 border-purple-300 dark:bg-purple-900/20 dark:border-purple-700';
      case 'knowledge': return 'bg-indigo-50 border-indigo-300 dark:bg-indigo-900/20 dark:border-indigo-700';
      case 'leanaide': return 'bg-cyan-50 border-cyan-300 dark:bg-cyan-900/20 dark:border-cyan-700';
      case 'crewai': return 'bg-orange-50 border-orange-300 dark:bg-orange-900/20 dark:border-orange-700';
      default: return 'bg-gray-50 border-gray-300 dark:bg-gray-800 dark:border-gray-700';
    }
  };

  return (
    <div className={`px-4 py-2 shadow-md rounded-md border-2 ${getNodeColor(data.type)} min-w-[150px]`}>
      <div className="flex items-center gap-2">
        {getNodeIcon(data.type)}
        <div className="font-bold text-gray-900 dark:text-white text-sm">{data.label}</div>
      </div>
    </div>
  );
};

// Mock data interfaces - these would come from the actual OpenEvolve API
interface WorkflowNode {
  id: string;
  type: 'start' | 'end' | 'evolution' | 'adversarial' | 'decomposition' | 'knowledge' | 'leanaide' | 'crewai' | 'custom';
  position: { x: number; y: number };
  data: {
    label: string;
    config?: any;
  };
}

interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  type: string;
}

interface WorkflowDefinition {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  createdAt: Date;
  updatedAt: Date;
  status: 'draft' | 'published' | 'archived';
}

const WorkflowBuilder: React.FC = () => {
  const [activeTab, setActiveTab] = useState('design');

  // Define node types
  const nodeTypes: NodeTypes = {
    custom: CustomNode,
  };

  // Convert workflow nodes to React Flow nodes
  const convertToFlowNodes = (workflow: WorkflowDefinition) => {
    return workflow.nodes.map(node => ({
      id: node.id,
      type: 'custom',
      position: node.position,
      data: {
        label: node.data.label,
        type: node.type
      }
    }));
  };

  // Convert workflow edges to React Flow edges
  const convertToFlowEdges = (workflow: WorkflowDefinition) => {
    return workflow.edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated: true,
      markerEnd: {
        type: MarkerType.ArrowClosed,
      }
    }));
  };

  const [workflows, setWorkflows] = useState<WorkflowDefinition[]>([
    {
      id: 'wf-1',
      name: 'Evolution Pipeline',
      description: 'Standard evolutionary algorithm pipeline',
      nodes: [
        { id: '1', type: 'start', position: { x: 100, y: 100 }, data: { label: 'Start' } },
        { id: '2', type: 'evolution', position: { x: 300, y: 100 }, data: { label: 'Evolution Engine' } },
        { id: '3', type: 'end', position: { x: 500, y: 100 }, data: { label: 'End' } }
      ],
      edges: [
        { id: 'e1-2', source: '1', target: '2', type: 'smoothstep' },
        { id: 'e2-3', source: '2', target: '3', type: 'smoothstep' }
      ],
      createdAt: new Date(Date.now() - 86400000 * 7), // 7 days ago
      updatedAt: new Date(Date.now() - 3600000 * 2), // 2 hours ago
      status: 'published'
    },
    {
      id: 'wf-2',
      name: 'Adversarial Validation',
      description: 'Red team/blue team validation pipeline',
      nodes: [
        { id: '1', type: 'start', position: { x: 100, y: 100 }, data: { label: 'Start' } },
        { id: '2', type: 'adversarial', position: { x: 300, y: 100 }, data: { label: 'Adversarial Testing' } },
        { id: '3', type: 'end', position: { x: 500, y: 100 }, data: { label: 'End' } }
      ],
      edges: [
        { id: 'e1-2', source: '1', target: '2', type: 'smoothstep' },
        { id: 'e2-3', source: '2', target: '3', type: 'smoothstep' }
      ],
      createdAt: new Date(Date.now() - 86400000 * 3), // 3 days ago
      updatedAt: new Date(Date.now() - 3600000 * 1), // 1 hour ago
      status: 'published'
    }
  ]);
  
  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowDefinition | null>(null);
  const [newWorkflow, setNewWorkflow] = useState({
    name: '',
    description: ''
  });
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleCreateWorkflow = () => {
    if (!newWorkflow.name.trim()) {
      alert('Workflow name is required');
      return;
    }

    const newWorkflowObj: WorkflowDefinition = {
      id: `wf-${Date.now()}`,
      name: newWorkflow.name,
      description: newWorkflow.description,
      nodes: [
        { id: '1', type: 'start', position: { x: 100, y: 100 }, data: { label: 'Start' } },
        { id: '2', type: 'end', position: { x: 300, y: 100 }, data: { label: 'End' } }
      ],
      edges: [
        { id: 'e1-2', source: '1', target: '2', type: 'smoothstep' }
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
      status: 'draft'
    };

    setWorkflows([...workflows, newWorkflowObj]);
    setNewWorkflow({ name: '', description: '' });
    setIsCreating(false);
  };

  const handleDeleteWorkflow = (workflowId: string) => {
    setWorkflows(workflows.filter(wf => wf.id !== workflowId));
    if (selectedWorkflow?.id === workflowId) {
      setSelectedWorkflow(null);
    }
  };

  const handlePublishWorkflow = (workflowId: string) => {
    setWorkflows(workflows.map(wf => 
      wf.id === workflowId ? { ...wf, status: 'published', updatedAt: new Date() } : wf
    ));
  };

  const handleUnpublishWorkflow = (workflowId: string) => {
    setWorkflows(workflows.map(wf => 
      wf.id === workflowId ? { ...wf, status: 'draft', updatedAt: new Date() } : wf
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'bg-green-500';
      case 'draft': return 'bg-yellow-500';
      case 'archived': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'published': return 'Published';
      case 'draft': return 'Draft';
      case 'archived': return 'Archived';
      default: return status;
    }
  };

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'start': return <GitBranch className="w-4 h-4 text-green-500" />;
      case 'end': return <GitBranch className="w-4 h-4 text-red-500" />;
      case 'evolution': return <Zap className="w-4 h-4 text-blue-500" />;
      case 'adversarial': return <Shield className="w-4 h-4 text-red-500" />;
      case 'decomposition': return <Target className="w-4 h-4 text-purple-500" />;
      case 'knowledge': return <BookOpen className="w-4 h-4 text-indigo-500" />;
      case 'leanaide': return <Database className="w-4 h-4 text-cyan-500" />;
      case 'crewai': return <Users className="w-4 h-4 text-orange-500" />;
      default: return <Layout className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="workflow-builder p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Workflow className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Workflow Builder
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
          Design and manage OpenEvolve workflow pipelines
        </p>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="design" label="Design">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Workflow List */}
            <div className="lg:col-span-1">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Workflows
                </h2>
                <BubbleButton
                  variant="default"
                  size="sm"
                  onClick={() => setIsCreating(true)}
                  className="flex items-center"
                >
                  <Plus className="w-4 h-4 mr-1" />
                  New
                </BubbleButton>
              </div>
              
              {isCreating ? (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-4">
                  <h3 className="text-md font-semibold text-gray-900 dark:text-white mb-3">
                    Create New Workflow
                  </h3>
                  
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Name
                      </label>
                      <BubbleInput
                        type="text"
                        value={newWorkflow.name}
                        onChange={(e) => setNewWorkflow({...newWorkflow, name: e.target.value})}
                        placeholder="Enter workflow name"
                        className="w-full"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Description
                      </label>
                      <textarea
                        value={newWorkflow.description}
                        onChange={(e) => setNewWorkflow({...newWorkflow, description: e.target.value})}
                        rows={3}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="Enter workflow description"
                      />
                    </div>
                    
                    <div className="flex space-x-2 pt-2">
                      <BubbleButton
                        variant="default"
                        size="sm"
                        onClick={handleCreateWorkflow}
                        className="flex-1"
                      >
                        Create
                      </BubbleButton>
                      <BubbleButton
                        variant="outline"
                        size="sm"
                        onClick={() => setIsCreating(false)}
                        className="flex-1"
                      >
                        Cancel
                      </BubbleButton>
                    </div>
                  </div>
                </div>
              ) : null}
              
              <div className="space-y-3">
                {workflows.map(workflow => (
                  <div 
                    key={workflow.id}
                    className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                      selectedWorkflow?.id === workflow.id 
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                        : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                    }`}
                    onClick={() => setSelectedWorkflow(workflow)}
                  >
                    <div className="flex justify-between items-start">
                      <h3 className="font-medium text-gray-900 dark:text-white truncate">
                        {workflow.name}
                      </h3>
                      <span className={`inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full ${
                        workflow.status === 'published' 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                          : workflow.status === 'draft' 
                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' 
                            : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                      }`}>
                        {getStatusText(workflow.status)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 truncate">
                      {workflow.description}
                    </p>
                    <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
                      <span>{workflow.nodes.length} nodes</span>
                      <span>{workflow.createdAt.toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Workflow Canvas */}
            <div className="lg:col-span-3">
              {selectedWorkflow ? (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                        {selectedWorkflow.name}
                      </h2>
                      <p className="text-gray-600 dark:text-gray-400">
                        {selectedWorkflow.description}
                      </p>
                    </div>
                    <div className="flex space-x-2">
                      <BubbleButton
                        variant="outline"
                        size="sm"
                        className="flex items-center"
                      >
                        <Play className="w-4 h-4 mr-1" />
                        Run
                      </BubbleButton>
                      <BubbleButton
                        variant="outline"
                        size="sm"
                        className="flex items-center"
                      >
                        <Edit3 className="w-4 h-4 mr-1" />
                        Edit
                      </BubbleButton>
                      <BubbleButton
                        variant="outline"
                        size="sm"
                        className="flex items-center"
                      >
                        <Trash2 className="w-4 h-4 mr-1" />
                        Delete
                      </BubbleButton>
                    </div>
                  </div>
                  
                  {/* Workflow Visualization */}
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg h-96 overflow-hidden">
                    {selectedWorkflow ? (
                      <ReactFlow
                        nodes={convertToFlowNodes(selectedWorkflow)}
                        edges={convertToFlowEdges(selectedWorkflow)}
                        nodeTypes={nodeTypes}
                        fitView
                        className="bg-gray-50 dark:bg-gray-900"
                      >
                        <Background color="#aaa" gap={16} />
                        <Controls />
                        <MiniMap
                          nodeColor={(node) => {
                            switch (node.data.type) {
                              case 'start': return '#10B981';
                              case 'end': return '#EF4444';
                              case 'evolution': return '#3B82F6';
                              case 'adversarial': return '#EF4444';
                              case 'decomposition': return '#8B5CF6';
                              case 'knowledge': return '#6366F1';
                              case 'leanaide': return '#06B6D4';
                              case 'crewai': return '#F97316';
                              default: return '#6B7280';
                            }
                          }}
                          className="dark:bg-gray-800"
                        />
                      </ReactFlow>
                    ) : (
                      <div className="h-full flex items-center justify-center">
                        <div className="text-center">
                          <Layout className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-600" />
                          <p className="mt-2 text-gray-500 dark:text-gray-400">
                            Select a workflow to visualize
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Node Palette */}
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                      Available Nodes
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <GitBranch className="w-6 h-6 text-green-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">Start</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <GitBranch className="w-6 h-6 text-red-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">End</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <Zap className="w-6 h-6 text-blue-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">Evolution</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <Shield className="w-6 h-6 text-red-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">Adversarial</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <Target className="w-6 h-6 text-purple-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">Decomposition</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <BookOpen className="w-6 h-6 text-indigo-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">Knowledge</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <Database className="w-6 h-6 text-cyan-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">LeanAide</span>
                      </div>
                      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-center mb-2">
                          <Users className="w-6 h-6 text-orange-500" />
                        </div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">CrewAI</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-12 text-center">
                  <Workflow className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-600" />
                  <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-white">
                    Select a workflow to edit
                  </h3>
                  <p className="mt-2 text-gray-500 dark:text-gray-400">
                    Choose a workflow from the list to view and edit its design
                  </p>
                  <div className="mt-6">
                    <BubbleButton
                      variant="default"
                      onClick={() => setIsCreating(true)}
                      className="flex items-center justify-center"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Create New Workflow
                    </BubbleButton>
                  </div>
                </div>
              )}
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="instances" label="Instances">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Workflow</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Progress</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {workflows.filter(wf => wf.status === 'published').map(workflow => (
                  <tr key={workflow.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{workflow.name}</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">{workflow.description}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      }`}>
                        Ready
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: '0%' }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900 dark:text-white">0%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      -
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <BubbleButton
                          variant="default"
                          size="sm"
                          className="flex items-center"
                        >
                          <Play className="w-4 h-4 mr-1" />
                          Run
                        </BubbleButton>
                        <button className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="templates" label="Templates">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center mb-4">
                <Zap className="w-8 h-8 text-blue-500 mr-3" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Evolution Pipeline
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Standard evolutionary algorithm pipeline for optimization problems
              </p>
              <BubbleButton
                variant="default"
                className="w-full"
              >
                Use Template
              </BubbleButton>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center mb-4">
                <Shield className="w-8 h-8 text-red-500 mr-3" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Adversarial Validation
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Red team/blue team validation pipeline for robustness testing
              </p>
              <BubbleButton
                variant="default"
                className="w-full"
              >
                Use Template
              </BubbleButton>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center mb-4">
                <Target className="w-8 h-8 text-purple-500 mr-3" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Decomposition Analysis
                </h3>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Problem decomposition and solution synthesis pipeline
              </p>
              <BubbleButton
                variant="default"
                className="w-full"
              >
                Use Template
              </BubbleButton>
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
          onClick={() => setIsCreating(true)}
          className="flex items-center"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Workflow
        </BubbleButton>
      </div>
    </div>
  );
};

export default WorkflowBuilder;
