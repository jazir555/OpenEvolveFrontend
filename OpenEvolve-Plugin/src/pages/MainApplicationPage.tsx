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
  Bell,
  User,
  LogOut,
  Menu,
  X,
  ChevronDown,
  ChevronRight,
  Collapse,
  Expand
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';
import Sidebar from '../components/Sidebar';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
  read: boolean;
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
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [notifications, setNotifications] = useState<Notification[]>([
    {
      id: 'notif-1',
      title: 'Workflow Completed',
      message: 'Evolution workflow #123 has completed successfully',
      type: 'success',
      timestamp: new Date(Date.now() - 300000),
      read: false
    },
    {
      id: 'notif-2',
      title: 'Adversarial Test Failed',
      message: 'Red team found vulnerabilities in solution #456',
      type: 'warning',
      timestamp: new Date(Date.now() - 600000),
      read: false
    },
    {
      id: 'notif-3',
      title: 'System Alert',
      message: 'High memory usage detected on worker node #2',
      type: 'error',
      timestamp: new Date(Date.now() - 900000),
      read: true
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
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  const unreadNotifications = notifications.filter(n => !n.read).length;

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const toggleNotifications = () => {
    setShowNotifications(!showNotifications);
  };

  const markNotificationAsRead = (id: string) => {
    setNotifications(notifications.map(n => 
      n.id === id ? { ...n, read: true } : n
    ));
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map(n => ({ ...n, read: true })));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      case 'degraded': return 'bg-yellow-500';
      case 'connected': return 'bg-green-500';
      case 'disconnected': return 'bg-red-500';
      case 'slow': return 'bg-yellow-500';
      case 'healthy': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'online': return 'Online';
      case 'offline': return 'Offline';
      case 'degraded': return 'Degraded';
      case 'connected': return 'Connected';
      case 'disconnected': return 'Disconnected';
      case 'slow': return 'Slow';
      case 'healthy': return 'Healthy';
      case 'error': return 'Error';
      default: return status;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading OpenEvolve application...</span>
      </div>
    );
  }

  return (
    <div className="main-application flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-16'} transition-all duration-300 ease-in-out flex-shrink-0`}>
        <Sidebar onNavigate={(route) => {
          // Handle navigation in the main app
          window.location.hash = route;
        }} />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Navigation Bar */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 z-10">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center">
              {/* Mobile menu button */}
              <button
                className="md:hidden mr-4 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => setSidebarOpen(!sidebarOpen)}
              >
                {sidebarOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
              
              {/* Search Bar */}
              <div className="relative hidden md:block">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <BubbleInput
                  type="text"
                  placeholder="Search workflows, components, knowledge..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-80"
                />
              </div>
            </div>

            {/* Right side controls */}
            <div className="flex items-center space-x-4">
              {/* System Status Indicators */}
              <div className="hidden md:flex items-center space-x-4 text-sm">
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-1 ${getStatusColor(systemStatus.apiStatus)}`}></div>
                  <span className="text-gray-600 dark:text-gray-400">API</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-1 ${getStatusColor(systemStatus.databaseStatus)}`}></div>
                  <span className="text-gray-600 dark:text-gray-400">DB</span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-1 ${getStatusColor(systemStatus.cacheStatus)}`}></div>
                  <span className="text-gray-600 dark:text-gray-400">Cache</span>
                </div>
              </div>

              {/* Notifications */}
              <div className="relative">
                <button
                  className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 relative"
                  onClick={toggleNotifications}
                >
                  <Bell className="h-6 w-6" />
                  {unreadNotifications > 0 && (
                    <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500"></span>
                  )}
                </button>
                
                {showNotifications && (
                  <div className="origin-top-right absolute right-0 mt-2 w-80 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 z-50">
                    <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">Notifications</h3>
                      <button 
                        className="text-sm text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                        onClick={markAllAsRead}
                      >
                        Mark all as read
                      </button>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {notifications.length > 0 ? (
                        notifications.map(notification => (
                          <div 
                            key={notification.id} 
                            className={`p-4 border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 ${
                              !notification.read ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                            }`}
                            onClick={() => markNotificationAsRead(notification.id)}
                          >
                            <div className="flex items-start">
                              <div className={`flex-shrink-0 mt-1 ${
                                notification.type === 'success' ? 'text-green-500' :
                                notification.type === 'warning' ? 'text-yellow-500' :
                                notification.type === 'error' ? 'text-red-500' : 'text-blue-500'
                              }`}>
                                {notification.type === 'success' ? <CheckCircle className="h-5 w-5" /> :
                                 notification.type === 'warning' ? <AlertTriangle className="h-5 w-5" /> :
                                 notification.type === 'error' ? <XCircle className="h-5 w-5" /> :
                                 <Info className="h-5 w-5" />}
                              </div>
                              <div className="ml-3 flex-1">
                                <p className="text-sm font-medium text-gray-900 dark:text-white">{notification.title}</p>
                                <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">{notification.message}</p>
                                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                                  {notification.timestamp.toLocaleTimeString()}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="p-6 text-center">
                          <Bell className="mx-auto h-12 w-12 text-gray-400" />
                          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No notifications</h3>
                          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                            You'll receive notifications here when events occur.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* User Profile */}
              <div className="flex items-center">
                <div className="ml-3 relative">
                  <div className="flex items-center">
                    <div className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                      <User className="w-5 h-5" />
                    </div>
                    <span className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300 hidden md:block">Admin</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
            <BubbleTab value="dashboard" label="Dashboard">
              <div className="mb-8">
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">OpenEvolve Dashboard</h1>
                    <p className="mt-2 text-gray-600 dark:text-gray-400">
                      AI-powered content improvement using evolutionary algorithms and adversarial testing
                    </p>
                  </div>
                  <BubbleButton
                    variant="default"
                    className="flex items-center"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    New Workflow
                  </BubbleButton>
                </div>
              </div>

              {/* Status Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <BubbleCard className="p-5">
                  <div className="flex items-center">
                    <Activity className="w-8 h-8 text-blue-500 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Workflows</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">{systemStatus.activeWorkflows}</p>
                    </div>
                  </div>
                </BubbleCard>

                <BubbleCard className="p-5">
                  <div className="flex items-center">
                    <Target className="w-8 h-8 text-green-500 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Completed Today</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">12</p>
                    </div>
                  </div>
                </BubbleCard>

                <BubbleCard className="p-5">
                  <div className="flex items-center">
                    <Shield className="w-8 h-8 text-red-500 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Failed Workflows</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">1</p>
                    </div>
                  </div>
                </BubbleCard>

                <BubbleCard className="p-5">
                  <div className="flex items-center">
                    <TrendingUp className="w-8 h-8 text-purple-500 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">92%</p>
                    </div>
                  </div>
                </BubbleCard>
              </div>

              {/* Quick Actions */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
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
                  onClick={() => setActiveTab('workflows')}
                >
                  <div className="flex items-center">
                    <Workflow className="w-8 h-8 text-blue-500 mr-3" />
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">Workflows</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Manage workflows</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  <Clock className="w-5 h-5 mr-2" />
                  Recent Activity
                </h2>
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Workflow</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Duration</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Updated</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">Optimization Run #1</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">ID: wf-001</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                            Completed
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">45 min</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">2 hours ago</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">Adversarial Test #1</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">ID: wf-002</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                            Running
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">23 min (running)</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">30 min ago</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">Decomposition Analysis #1</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">ID: wf-003</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                            Failed
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">12 min</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">1 hour ago</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </BubbleTab>

            <BubbleTab value="evolution" label="Evolution">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">Evolution Engine</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Configure and run evolutionary algorithms for optimization and improvement
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Population Size
                    </label>
                    <BubbleInput
                      type="number"
                      defaultValue="100"
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Generations
                    </label>
                    <BubbleInput
                      type="number"
                      defaultValue="50"
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Mutation Rate
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      defaultValue="0.1"
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Crossover Rate
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      defaultValue="0.8"
                      className="w-full"
                    />
                  </div>
                </div>
                
                <div className="mt-6">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Problem Statement
                  </label>
                  <textarea
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Enter your problem statement here..."
                  ></textarea>
                </div>
                
                <div className="mt-6 flex justify-end">
                  <BubbleButton
                    variant="default"
                    className="flex items-center"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Start Evolution
                  </BubbleButton>
                </div>
              </div>
            </BubbleTab>

            <BubbleTab value="adversarial" label="Adversarial">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">Adversarial Testing</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Test your solutions with adversarial methods to ensure robustness
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Attack Strength
                    </label>
                    <BubbleSelect className="w-full">
                      <option value="low">Low</option>
                      <option value="medium" selected>Medium</option>
                      <option value="high">High</option>
                      <option value="maximum">Maximum</option>
                    </BubbleSelect>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Number of Attacks
                    </label>
                    <BubbleInput
                      type="number"
                      defaultValue="10"
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Defense Strategy
                    </label>
                    <BubbleSelect className="w-full">
                      <option value="robust">Robust</option>
                      <option value="adaptive">Adaptive</option>
                      <option value="certified">Certified</option>
                      <option value="detection">Detection</option>
                    </BubbleSelect>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Robustness Threshold
                    </label>
                    <BubbleInput
                      type="number"
                      step="0.01"
                      defaultValue="0.7"
                      className="w-full"
                    />
                  </div>
                </div>
                
                <div className="mt-6">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Content to Test
                  </label>
                  <textarea
                    rows={6}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Enter content to test for adversarial robustness..."
                  ></textarea>
                </div>
                
                <div className="mt-6 flex justify-end space-x-3">
                  <BubbleButton
                    variant="outline"
                  >
                    Preview Attack
                  </BubbleButton>
                  <BubbleButton
                    variant="default"
                    className="flex items-center"
                  >
                    <Shield className="w-4 h-4 mr-2" />
                    Start Adversarial Test
                  </BubbleButton>
                </div>
              </div>
            </BubbleTab>

            <BubbleTab value="workflows" label="Workflows">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Workflows</h2>
                  <BubbleButton
                    variant="default"
                    className="flex items-center"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    New Workflow
                  </BubbleButton>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Progress</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Created</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">Optimization Pipeline #1</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">ID: wf-001</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                            Completed
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                              <div className="bg-green-600 h-2 rounded-full" style={{ width: '100%' }}></div>
                            </div>
                            <span className="text-sm text-gray-900 dark:text-white">100%</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">2 days ago</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300 mr-3">
                            View
                          </button>
                          <button className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300">
                            Logs
                          </button>
                        </td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">Adversarial Validation #1</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">ID: wf-002</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                            Running
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                              <div className="bg-blue-600 h-2 rounded-full" style={{ width: '65%' }}></div>
                            </div>
                            <span className="text-sm text-gray-900 dark:text-white">65%</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">1 day ago</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300 mr-3">
                            View
                          </button>
                          <button className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300">
                            Logs
                          </button>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </BubbleTab>
          </BubbleTabs>
        </main>

        {/* Footer */}
        <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 py-4 px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              © {new Date().getFullYear()} OpenEvolve - AI Evolution Platform
            </div>
            <div className="mt-2 md:mt-0 flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
              <span>v2.5.1</span>
              <span>•</span>
              <span>Build: 2026.01.01</span>
              <span>•</span>
              <span>Mode: Production</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default MainApplication;
