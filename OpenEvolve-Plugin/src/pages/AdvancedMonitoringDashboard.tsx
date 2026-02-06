/**
 * OpenEvolve Advanced Monitoring Dashboard Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based monitoring dashboard UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Play, 
  Square, 
  RotateCcw, 
  BarChart3, 
  Target, 
  Users,
  GitBranch,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  TrendingUp,
  Database,
  Zap,
  Shield,
  Eye,
  EyeOff,
  Search,
  Filter,
  Download,
  Upload,
  Settings,
  Info
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface MonitoringEvent {
  id: string;
  timestamp: Date;
  workflowId: string;
  stage: string;
  subProblemId?: string;
  gauntletName?: string;
  status: 'success' | 'failure' | 'warning' | 'info';
  message: string;
  metadata: Record<string, any>;
}

interface WorkflowSummary {
  activeWorkflows: number;
  completedWorkflows: number;
  failedWorkflows: number;
  activeTickets: number;
  completedTickets: number;
  failedTickets: number;
  totalGauntletRuns: number;
  successfulGauntletRuns: number;
  successRate: number;
}

interface GauntletPerformance {
  redTeam: {
    totalReports: number;
    approvalRate: number;
    avgScore: number;
  };
  goldTeam: {
    totalReports: number;
    approvalRate: number;
    avgScore: number;
  };
  totalReports: number;
  approvalRate: number;
}

interface SovereignWorkflow {
  workflowId: string;
  currentStage: string;
  crewaiWorkflowId?: string;
  progress: number;
  problemStatement: string;
  currentSubProblemId?: string;
  currentGauntletName?: string;
}

const AdvancedMonitoringDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [monitoringEvents, setMonitoringEvents] = useState<MonitoringEvent[]>([]);
  const [workflowSummary, setWorkflowSummary] = useState<WorkflowSummary>({
    activeWorkflows: 0,
    completedWorkflows: 0,
    failedWorkflows: 0,
    activeTickets: 0,
    completedTickets: 0,
    failedTickets: 0,
    totalGauntletRuns: 0,
    successfulGauntletRuns: 0,
    successRate: 0
  });
  const [gauntletPerformance, setGauntletPerformance] = useState<GauntletPerformance>({
    redTeam: { totalReports: 0, approvalRate: 0, avgScore: 0 },
    goldTeam: { totalReports: 0, approvalRate: 0, avgScore: 0 },
    totalReports: 0,
    approvalRate: 0
  });
  const [activeWorkflow, setActiveWorkflow] = useState<SovereignWorkflow | null>(null);
  const [monitoringStatus, setMonitoringStatus] = useState<'idle' | 'running' | 'stopped'>('idle');
  const [isLoading, setIsLoading] = useState(true);

  // Simulate loading data
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      setWorkflowSummary({
        activeWorkflows: 2,
        completedWorkflows: 15,
        failedWorkflows: 1,
        activeTickets: 8,
        completedTickets: 42,
        failedTickets: 3,
        totalGauntletRuns: 50,
        successfulGauntletRuns: 45,
        successRate: 90
      });
      
      setGauntletPerformance({
        redTeam: { totalReports: 25, approvalRate: 0.76, avgScore: 0.72 },
        goldTeam: { totalReports: 25, approvalRate: 0.84, avgScore: 0.81 },
        totalReports: 50,
        approvalRate: 0.80
      });
      
      setActiveWorkflow({
        workflowId: 'wf-001',
        currentStage: 'decomposition',
        crewaiWorkflowId: 'hwf-001',
        progress: 0.65,
        problemStatement: 'Optimize the complex mathematical function f(x,y) = x^2 + y^2 - 2xy + 3x - 4y subject to constraints...',
        currentSubProblemId: 'sub-001',
        currentGauntletName: 'Red Team Critique'
      });
      
      setMonitoringEvents([
        {
          id: 'evt-001',
          timestamp: new Date(Date.now() - 300000),
          workflowId: 'wf-001',
          stage: 'decomposition',
          status: 'success',
          message: 'Sub-problem 1.1 decomposed successfully',
          metadata: { subProblemId: 'sub-001' }
        },
        {
          id: 'evt-002',
          timestamp: new Date(Date.now() - 600000),
          workflowId: 'wf-001',
          stage: 'gauntlet',
          status: 'warning',
          message: 'Red team found potential vulnerability in solution',
          metadata: { gauntletName: 'Red Team Critique', subProblemId: 'sub-001' }
        },
        {
          id: 'evt-003',
          timestamp: new Date(Date.now() - 900000),
          workflowId: 'wf-002',
          stage: 'verification',
          status: 'success',
          message: 'Gold team verified solution quality',
          metadata: { gauntletName: 'Gold Team Verification' }
        }
      ]);
      
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const handleStartMonitoring = () => {
    setMonitoringStatus('running');
  };

  const handleStopMonitoring = () => {
    setMonitoringStatus('stopped');
  };

  const handleRefresh = () => {
    // Simulate refresh
    window.location.reload();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'bg-green-500';
      case 'failure': return 'bg-red-500';
      case 'warning': return 'bg-yellow-500';
      case 'info': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'success': return 'Success';
      case 'failure': return 'Failure';
      case 'warning': return 'Warning';
      case 'info': return 'Info';
      default: return status;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading monitoring dashboard...</span>
      </div>
    );
  }

  return (
    <div className="advanced-monitoring-dashboard p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Activity className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              SG-D Workflow Monitoring Dashboard
            </h1>
          </div>
          <div className="flex items-center space-x-3">
            <BubbleButton
              variant={monitoringStatus === 'running' ? 'destructive' : 'default'}
              size="sm"
              onClick={monitoringStatus === 'running' ? handleStopMonitoring : handleStartMonitoring}
              className="flex items-center"
            >
              {monitoringStatus === 'running' ? (
                <>
                  <Square className="w-4 h-4 mr-2" />
                  Stop Monitoring
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Start Monitoring
                </>
              )}
            </BubbleButton>
            <BubbleButton
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="flex items-center"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Refresh
            </BubbleButton>
          </div>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Monitor and analyze Sovereign-Grade Decomposition workflows
        </p>
      </div>

      {/* Status Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Activity className="w-8 h-8 text-blue-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Workflows</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{workflowSummary.activeWorkflows}</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Completed</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{workflowSummary.completedWorkflows}</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <XCircle className="w-8 h-8 text-red-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Failed</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{workflowSummary.failedWorkflows}</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <TrendingUp className="w-8 h-8 text-purple-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{workflowSummary.successRate}%</p>
            </div>
          </div>
        </BubbleCard>
      </div>

      {/* Main Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="overview" label="Overview">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Current Workflow Status */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <GitBranch className="w-5 h-5 mr-2" />
                Current Workflow Status
              </h2>
              
              {activeWorkflow ? (
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                  <div className="mb-4">
                    <h3 className="font-medium text-gray-900 dark:text-white">Workflow: {activeWorkflow.workflowId}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Current Stage: {activeWorkflow.currentStage}</p>
                  </div>
                  
                  {activeWorkflow.crewaiWorkflowId && (
                    <div className="mb-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400">CrewAI Workflow ID: {activeWorkflow.crewaiWorkflowId}</p>
                    </div>
                  )}
                  
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">Progress</span>
                      <span className="font-medium text-gray-900 dark:text-white">{(activeWorkflow.progress * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                      <div 
                        className="bg-blue-600 h-2.5 rounded-full" 
                        style={{ width: `${activeWorkflow.progress * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      <strong>Problem Statement:</strong> {activeWorkflow.problemStatement.substring(0, 100)}...
                    </p>
                  </div>
                  
                  {activeWorkflow.currentSubProblemId && (
                    <div className="mb-2">
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Currently Processing:</strong> {activeWorkflow.currentSubProblemId}
                      </p>
                    </div>
                  )}
                  
                  {activeWorkflow.currentGauntletName && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Current Gauntlet:</strong> {activeWorkflow.currentGauntletName}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-8 text-center">
                  <GitBranch className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-600" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No Active Workflow</h3>
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    Start a Sovereign-Grade workflow to begin monitoring
                  </p>
                </div>
              )}
            </div>

            {/* Gauntlet Performance */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                Gauntlet Performance Analysis
              </h2>
              
              <div className="space-y-6">
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="font-medium text-gray-900 dark:text-white">Overall Performance</h3>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {(gauntletPerformance.approvalRate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">{gauntletPerformance.totalReports}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Total Reports</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {(gauntletPerformance.approvalRate * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Approval Rate</p>
                    </div>
                  </div>
                </div>
                
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900 dark:text-white mb-3">Red Team (Critique) Performance</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <p className="text-xl font-bold text-gray-900 dark:text-white">{gauntletPerformance.redTeam.totalReports}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Total Reports</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xl font-bold text-gray-900 dark:text-white">
                        {(gauntletPerformance.redTeam.approvalRate * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Approval Rate</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xl font-bold text-gray-900 dark:text-white">
                        {gauntletPerformance.redTeam.avgScore.toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Avg Score</p>
                    </div>
                  </div>
                </div>
                
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900 dark:text-white mb-3">Gold Team (Verification) Performance</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <p className="text-xl font-bold text-gray-900 dark:text-white">{gauntletPerformance.goldTeam.totalReports}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Total Reports</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xl font-bold text-gray-900 dark:text-white">
                        {(gauntletPerformance.goldTeam.approvalRate * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Approval Rate</p>
                    </div>
                    <div className="text-center">
                      <p className="text-xl font-bold text-gray-900 dark:text-white">
                        {gauntletPerformance.goldTeam.avgScore.toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Avg Score</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="events" label="Monitoring Events">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Workflow</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Stage</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Message</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {monitoringEvents.map(event => (
                  <tr key={event.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {event.timestamp.toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                      {event.workflowId}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {event.stage}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        event.status === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                        event.status === 'failure' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        event.status === 'warning' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                      }`}>
                        {getStatusText(event.status)}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                      {event.message}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {monitoringEvents.length === 0 && (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              <Clock className="mx-auto h-12 w-12" />
              <h3 className="mt-2 text-sm font-medium">No monitoring events</h3>
              <p className="mt-1 text-sm">Events will appear as workflows are processed</p>
            </div>
          )}
        </BubbleTab>

        <BubbleTab value="integration" label="Integration">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <BubbleCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Database className="w-5 h-5 mr-2" />
                Integration Status
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <span className="text-gray-700 dark:text-gray-300">OpenEvolve API</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Connected
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <span className="text-gray-700 dark:text-gray-300">CrewAI API</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Connected
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <span className="text-gray-700 dark:text-gray-300">Gauntlet Service</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Available
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <span className="text-gray-700 dark:text-gray-300">Knowledge Engine</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Synced
                  </span>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Zap className="w-5 h-5 mr-2" />
                Performance Metrics
              </h3>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">API Response Time</span>
                    <span className="font-medium text-gray-900 dark:text-white">42ms</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-green-600 h-2 rounded-full" style={{ width: '42%' }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">Throughput</span>
                    <span className="font-medium text-gray-900 dark:text-white">120 req/min</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-blue-600 h-2 rounded-full" style={{ width: '75%' }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">Error Rate</span>
                    <span className="font-medium text-gray-900 dark:text-white">0.02%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-yellow-600 h-2 rounded-full" style={{ width: '2%' }}></div>
                  </div>
                </div>
              </div>
            </BubbleCard>
          </div>
        </BubbleTab>
      </BubbleTabs>

      {/* Action Buttons */}
      <div className="mt-8 flex justify-end space-x-3">
        <BubbleButton
          variant="outline"
          className="flex items-center"
          onClick={handleRefresh}
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Refresh
        </BubbleButton>
        <BubbleButton
          variant="default"
          className="flex items-center"
        >
          <Download className="w-4 h-4 mr-2" />
          Export Data
        </BubbleButton>
      </div>
    </div>
  );
};

export default AdvancedMonitoringDashboard;
