/**
 * OpenEvolve Sidebar Component for BubbleLab
 * 
 * This component replaces the BubbleLab UI-based sidebar UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState } from 'react';
import { 
  Home, 
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
  ChevronDown,
  ChevronRight,
  Collapse,
  Expand
} from 'lucide-react';

interface SidebarItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  route: string;
  children?: SidebarItem[];
}

const Sidebar: React.FC<{ onNavigate: (route: string) => void }> = ({ onNavigate }) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set(['workflows']));
  const [activeItem, setActiveItem] = useState('dashboard');

  const sidebarItems: SidebarItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <Home className="w-5 h-5" />,
      route: '/openevolve'
    },
    {
      id: 'workflows',
      label: 'Workflows',
      icon: <Workflow className="w-5 h-5" />,
      route: '/openevolve/workflows',
      children: [
        {
          id: 'create-workflow',
          label: 'Create Workflow',
          icon: <Plus className="w-4 h-4" />,
          route: '/openevolve/workflows/create'
        },
        {
          id: 'workflow-orchestrator',
          label: 'Orchestrator',
          icon: <GitBranch className="w-4 h-4" />,
          route: '/openevolve/orchestrator'
        },
        {
          id: 'workflow-templates',
          label: 'Templates',
          icon: <BookOpen className="w-4 h-4" />,
          route: '/openevolve/templates'
        }
      ]
    },
    {
      id: 'evolution',
      label: 'Evolution',
      icon: <Zap className="w-5 h-5" />,
      route: '/openevolve/evolution'
    },
    {
      id: 'adversarial',
      label: 'Adversarial',
      icon: <Shield className="w-5 h-5" />,
      route: '/openevolve/adversarial'
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <BarChart3 className="w-5 h-5" />,
      route: '/openevolve/analytics'
    },
    {
      id: 'knowledge',
      label: 'Knowledge Base',
      icon: <BookOpen className="w-5 h-5" />,
      route: '/openevolve/knowledge'
    },
    {
      id: 'monitoring',
      label: 'Monitoring',
      icon: <Activity className="w-5 h-5" />,
      route: '/openevolve/monitoring'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Settings className="w-5 h-5" />,
      route: '/openevolve/settings'
    }
  ];

  const toggleExpand = (itemId: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(itemId)) {
      newExpanded.delete(itemId);
    } else {
      newExpanded.add(itemId);
    }
    setExpandedItems(newExpanded);
  };

  const handleItemClick = (item: SidebarItem) => {
    setActiveItem(item.id);
    if (item.route) {
      onNavigate(item.route);
    }
  };

  const renderSidebarItem = (item: SidebarItem) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.has(item.id);
    
    return (
      <div key={item.id} className="mb-1">
        <div
          className={`flex items-center px-3 py-2 text-sm font-medium rounded-md cursor-pointer transition-colors ${
            activeItem === item.id
              ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
              : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
          }`}
          onClick={() => hasChildren ? toggleExpand(item.id) : handleItemClick(item)}
        >
          <span className="mr-3">{item.icon}</span>
          <span className="flex-1">{item.label}</span>
          {hasChildren && (
            <span>
              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </span>
          )}
        </div>
        
        {hasChildren && isExpanded && (
          <div className="ml-6 mt-1 space-y-1">
            {item.children!.map(child => (
              <div
                key={child.id}
                className={`flex items-center px-3 py-2 text-sm font-medium rounded-md cursor-pointer transition-colors ${
                  activeItem === child.id
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-800/50 dark:text-blue-300'
                    : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'
                }`}
                onClick={() => handleItemClick(child)}
              >
                <span className="mr-3">{child.icon}</span>
                <span>{child.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="sidebar w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-full flex flex-col">
      {/* Logo/Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center">
          <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">OpenEvolve</h1>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">AI Evolution Platform</p>
      </div>

      {/* Search */}
      <div className="p-4">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search..."
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md leading-5 bg-white dark:bg-gray-700 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 overflow-y-auto px-2 py-4">
        {sidebarItems.map(renderSidebarItem)}
      </div>

      {/* User Profile Section */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className="h-10 w-10 rounded-full bg-blue-500 flex items-center justify-center text-white">
              <User className="w-6 h-6" />
            </div>
          </div>
          <div className="ml-3">
            <p className="text-sm font-medium text-gray-900 dark:text-white">Admin User</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">Administrator</p>
          </div>
          <button className="ml-auto p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
            <Bell className="h-5 w-5" />
          </button>
        </div>
        
        <div className="mt-4 space-y-1">
          <button
            className="w-full flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <Settings className="w-4 h-4 mr-3" />
            Settings
          </button>
          <button
            className="w-full flex items-center px-3 py-2 text-sm font-medium text-red-600 dark:text-red-400 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20"
          >
            <LogOut className="w-4 h-4 mr-3" />
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
