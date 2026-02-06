/**
 * OpenEvolve Main Layout Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based main layout
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Workflow, 
  BarChart3, 
  Settings, 
  Brain, 
  Zap, 
  Shield, 
  GitBranch, 
  BookOpen, 
  Database, 
  Users, 
  Target, 
  Activity, 
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

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
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

  const unreadNotifications = notifications.filter(n => !n.read).length;

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

  return (
    <div className="main-layout flex h-screen bg-gray-50 dark:bg-gray-900">
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
          {children}
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

export default MainLayout;
