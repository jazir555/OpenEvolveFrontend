/**
 * OpenEvolve Main Dashboard Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based main dashboard UI
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
  Clock,
  Users,
  Target,
  TrendingUp,
  GitBranch,
  BookOpen,
  Database
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface SystemMetrics {
  activeWorkflows: number;
  totalWorkflows: number;
  successRate: number;
  avgExecutionTime: number;
  cpuUsage: number;
  memoryUsage: number;
  totalKnowledgeEntries: number;
  activeEvolutionRuns: number;
  completedAdversarialTests: number;
}

interface RecentActivity {
  id: string;
  action: string;
  timestamp: Date;
  status: 'success' | 'warning' | 'error';
  details: string;
}

interface QuickAction {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  route: string;
}

const OpenEvolveDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    activeWorkflows: 0,
    totalWorkflows: 0,
    successRate: 0,
    avgExecutionTime: 0,
    cpuUsage: 0,
    memoryUsage: 0,
    totalKnowledgeEntries: 0,
    activeEvolutionRuns: 0,
    completedAdversarialTests: 0
  });
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Quick actions for the dashboard
  const quickActions: QuickAction[] = [
    {
      id: 'create-workflow',
      title: 'Create Workflow',
      description: 'Design a new OpenEvolve workflow',
      icon: <GitBranch className="w-6 h-6 text-blue-500" />,
      route: '/openevolve/workflows'
    },
    {
      id: 'run-evolution',
      title: 'Run Evolution',
      description: 'Start an evolutionary algorithm',
      icon: <Zap className="w-6 h-6 text-yellow-500" />,
      route: '/openevolve/evolution'
    },
    {
      id: 'adversarial-test',
      title: 'Adversarial Test',
      description: 'Run adversarial validation',
      icon: <Shield className="w-6 h-6 text-red-500" />,
      route: '/openevolve/adversarial'
    },
    {
      id: 'knowledge-base',
      title: 'Knowledge Base',
      description: 'Access knowledge resources',
      icon: <BookOpen className="w-6 h-6 text-indigo-500" />,
      route: '/openevolve/knowledge'
    }
  ];

  // Simulate loading data
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      setSystemMetrics({
        activeWorkflows: 3,
        totalWorkflows: 42,
        successRate: 87.5,
        avgExecutionTime: 1245,
        cpuUsage: 42,
        memoryUsage: 68,
        totalKnowledgeEntries: 128,
        activeEvolutionRuns: 2,
        completedAdversarialTests: 15
      });
      
      setRecentActivity([
        {
          id: 'act-001',
          action: 'Workflow Started',
          timestamp: new Date(Date.now() - 300000),
          status: 'success',
          details: 'Evolution workflow wf-001 initiated'
        },
        {
          id: 'act-002',
          action: 'Decomposition Complete',
          timestamp: new Date(Date.now() - 600000),
          status: 'success',
          details: 'Problem decomposition completed successfully'
        },
        {
          id: 'act-003',
          action: 'Adversarial Test Failed',
          timestamp: new Date(Date.now() - 1200000),
          status: 'warning',
          details: 'One test case failed validation'
        },
        {
          id: 'act-004',
          action: 'Knowledge Entry Added',
          timestamp: new Date(Date.now() - 1800000),
          status: 'success',
          details: 'New algorithm documentation added'
        }
      ]);
      
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const getActivityIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading dashboard...</span>
      </div>
    );
  }

  return (
    <div className="openevolve-dashboard p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve Dashboard
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
          Monitor and manage your OpenEvolve workflows and system performance
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {quickActions.map(action => (
          <div 
            key={action.id} 
            className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => window.location.hash = action.route}
          >
            <div className="flex items-center">
              {action.icon}
              <div className="ml-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">{action.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{action.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Activity className="w-8 h-8 text-blue-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Workflows</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{systemMetrics.activeWorkflows}</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Target className="w-8 h-8 text-green-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{systemMetrics.successRate}%</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <TrendingUp className="w-8 h-8 text-purple-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg. Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{systemMetrics.avgExecutionTime}s</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Database className="w-8 h-8 text-cyan-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Knowledge</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{systemMetrics.totalKnowledgeEntries}</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <GitBranch className="w-8 h-8 text-yellow-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Workflows</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{systemMetrics.totalWorkflows}</p>
            </div>
          </div>
        </BubbleCard>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="overview" label="Overview">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Health */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                System Health
              </h2>
              <div className="space-y-4">
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600 dark:text-gray-400">CPU Usage</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.cpuUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${systemMetrics.cpuUsage}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600 dark:text-gray-400">Memory Usage</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.memoryUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${systemMetrics.memoryUsage}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600 dark:text-gray-400">Active Evolution Runs</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.activeEvolutionRuns}</span>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Currently running evolutionary algorithms
                  </div>
                </div>
                
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600 dark:text-gray-400">Completed Adversarial Tests</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.completedAdversarialTests}</span>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Successfully completed adversarial validation tests
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2" />
                Recent Activity
              </h2>
              <div className="space-y-4">
                {recentActivity.map(activity => (
                  <div 
                    key={activity.id} 
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex items-start">
                      {getActivityIcon(activity.status)}
                      <div className="ml-3 flex-1">
                        <div className="flex justify-between">
                          <h3 className="font-medium text-gray-900 dark:text-white">{activity.action}</h3>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {activity.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{activity.details}</p>
                      </div>
                    </div>
                  </div>
                ))}
                
                {recentActivity.length === 0 && (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    No recent activity
                  </div>
                )}
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="workflows" label="Workflows">
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
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Evolution Pipeline #1</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Standard optimization workflow</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                      Running
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full" 
                          style={{ width: '65%' }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-900 dark:text-white">65%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    2h 30m (running)
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300">
                        View
                      </button>
                      <button className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300">
                        Logs
                      </button>
                    </div>
                  </td>
                </tr>
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">Adversarial Test #2</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Red team validation</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      Completed
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                        <div 
                          className="bg-green-600 h-2 rounded-full" 
                          style={{ width: '100%' }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-900 dark:text-white">100%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    1h 15m
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300">
                        View
                      </button>
                      <button className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300">
                        Logs
                      </button>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="analytics" label="Analytics">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <BubbleCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Performance Metrics
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">CPU Usage</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.cpuUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${systemMetrics.cpuUsage}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">Memory Usage</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.memoryUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${systemMetrics.memoryUsage}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">Success Rate</span>
                    <span className="font-medium text-gray-900 dark:text-white">{systemMetrics.successRate}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full" 
                      style={{ width: `${systemMetrics.successRate}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                System Status
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">API Status</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Online
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Database</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Connected
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Cache</span>
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full dark:bg-green-900 dark:text-green-200">
                    Healthy
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Workers</span>
                  <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full dark:bg-blue-900 dark:text-blue-200">
                    4 Active
                  </span>
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
    </div>
  );
};

export default OpenEvolveDashboard;
