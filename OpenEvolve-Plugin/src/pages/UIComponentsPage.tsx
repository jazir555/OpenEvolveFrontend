/**
 * OpenEvolve UI Components Page for BubbleLab
 * 
 * This component replaces the Streamlit-based UI components page
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Settings, 
  Search, 
  Plus, 
  Edit3, 
  Trash2, 
  Download, 
  Upload, 
  Filter, 
  Eye, 
  EyeOff, 
  BarChart3, 
  Users, 
  GitBranch, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Tag, 
  Folder, 
  FileText, 
  Database as DatabaseIcon,
  Star,
  Heart,
  MessageCircle,
  Share2
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleInput, BubbleSelect, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface UIComponent {
  id: string;
  name: string;
  description: string;
  category: string;
  usageCount: number;
  lastUsed: Date;
  status: 'active' | 'deprecated' | 'experimental';
}

interface ComponentParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'select' | 'array';
  defaultValue: any;
  required: boolean;
  description: string;
}

const UIComponentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('components');
  const [components, setComponents] = useState<UIComponent[]>([
    {
      id: 'comp-1',
      name: 'BubbleButton',
      description: 'Custom button component with loading states',
      category: 'inputs',
      usageCount: 128,
      lastUsed: new Date(Date.now() - 3600000),
      status: 'active'
    },
    {
      id: 'comp-2',
      name: 'BubbleCard',
      description: 'Container component for grouping content',
      category: 'layout',
      usageCount: 95,
      lastUsed: new Date(Date.now() - 7200000),
      status: 'active'
    },
    {
      id: 'comp-3',
      name: 'BubbleInput',
      description: 'Form input with validation',
      category: 'inputs',
      usageCount: 142,
      lastUsed: new Date(Date.now() - 1800000),
      status: 'active'
    },
    {
      id: 'comp-4',
      name: 'BubbleSelect',
      description: 'Dropdown selector component',
      category: 'inputs',
      usageCount: 87,
      lastUsed: new Date(Date.now() - 10800000),
      status: 'active'
    },
    {
      id: 'comp-5',
      name: 'BubbleTabs',
      description: 'Tab navigation system',
      category: 'navigation',
      usageCount: 76,
      lastUsed: new Date(Date.now() - 14400000),
      status: 'active'
    },
    {
      id: 'comp-6',
      name: 'BubbleTab',
      description: 'Individual tab component',
      category: 'navigation',
      usageCount: 76,
      lastUsed: new Date(Date.now() - 14400000),
      status: 'active'
    }
  ]);
  
  const [selectedComponent, setSelectedComponent] = useState<UIComponent | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isLoading, setIsLoading] = useState(true);

  // Get unique categories
  const categories = Array.from(new Set(components.map(comp => comp.category)));

  // Filter components based on search and category
  const filteredComponents = components.filter(comp => {
    const matchesSearch = comp.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         comp.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || comp.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

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

  const handleRefresh = () => {
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'deprecated': return 'bg-red-500';
      case 'experimental': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return 'Active';
      case 'deprecated': return 'Deprecated';
      case 'experimental': return 'Experimental';
      default: return status;
    }
  };

  const getComponentParameters = (componentId: string): ComponentParameter[] => {
    switch (componentId) {
      case 'comp-1': // BubbleButton
        return [
          { name: 'variant', type: 'select', defaultValue: 'default', required: true, description: 'Button style variant' },
          { name: 'size', type: 'select', defaultValue: 'md', required: false, description: 'Button size' },
          { name: 'isLoading', type: 'boolean', defaultValue: false, required: false, description: 'Show loading indicator' },
          { name: 'disabled', type: 'boolean', defaultValue: false, required: false, description: 'Disable button' }
        ];
      case 'comp-2': // BubbleCard
        return [
          { name: 'children', type: 'string', defaultValue: '', required: true, description: 'Card content' },
          { name: 'className', type: 'string', defaultValue: '', required: false, description: 'Additional CSS classes' }
        ];
      case 'comp-3': // BubbleInput
        return [
          { name: 'type', type: 'select', defaultValue: 'text', required: true, description: 'Input type' },
          { name: 'label', type: 'string', defaultValue: '', required: false, description: 'Input label' },
          { name: 'error', type: 'string', defaultValue: '', required: false, description: 'Error message' },
          { name: 'className', type: 'string', defaultValue: '', required: false, description: 'Additional CSS classes' }
        ];
      case 'comp-4': // BubbleSelect
        return [
          { name: 'options', type: 'array', defaultValue: [], required: true, description: 'Select options' },
          { name: 'label', type: 'string', defaultValue: '', required: false, description: 'Select label' },
          { name: 'error', type: 'string', defaultValue: '', required: false, description: 'Error message' },
          { name: 'className', type: 'string', defaultValue: '', required: false, description: 'Additional CSS classes' }
        ];
      case 'comp-5': // BubbleTabs
        return [
          { name: 'value', type: 'string', defaultValue: '', required: true, description: 'Current active tab' },
          { name: 'onValueChange', type: 'string', defaultValue: '', required: true, description: 'Callback when tab changes' },
          { name: 'children', type: 'string', defaultValue: '', required: true, description: 'Tab components' }
        ];
      case 'comp-6': // BubbleTab
        return [
          { name: 'value', type: 'string', defaultValue: '', required: true, description: 'Tab identifier' },
          { name: 'label', type: 'string', defaultValue: '', required: true, description: 'Tab label' },
          { name: 'children', type: 'string', defaultValue: '', required: true, description: 'Tab content' }
        ];
      default:
        return [];
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading UI components...</span>
      </div>
    );
  }

  return (
    <div className="ui-components-page p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <DatabaseIcon className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              OpenEvolve UI Components
            </h1>
          </div>
          <div className="flex items-center space-x-3">
            <BubbleButton
              variant="outline"
              size="sm"
              className="flex items-center"
              onClick={handleRefresh}
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Refresh
            </BubbleButton>
            <BubbleButton
              variant="default"
              size="sm"
              className="flex items-center"
            >
              <Plus className="w-4 h-4 mr-2" />
              New Component
            </BubbleButton>
          </div>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          React-based UI components for the OpenEvolve BubbleLab plugin system
        </p>
      </div>

      {/* Search and Filters */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <BubbleInput
              type="text"
              placeholder="Search components..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-full"
            />
          </div>
        </div>
        
        <div>
          <BubbleSelect
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full"
          >
            <option value="all">All Categories</option>
            {categories.map(category => (
              <option key={category} value={category}>
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </option>
            ))}
          </BubbleSelect>
        </div>
        
        <div className="flex space-x-2">
          <BubbleButton
            variant="outline"
            size="sm"
            className="flex items-center justify-center"
          >
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </BubbleButton>
          <BubbleButton
            variant="outline"
            size="sm"
            className="flex items-center justify-center"
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </BubbleButton>
        </div>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="components" label="Components">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredComponents.map(component => (
              <div 
                key={component.id} 
                className={`border rounded-lg p-5 cursor-pointer transition-all ${
                  selectedComponent?.id === component.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-md'
                    : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
                onClick={() => setSelectedComponent(component)}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">{component.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{component.description}</p>
                  </div>
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(component.status)}`}></div>
                </div>
                <div className="mt-4 flex justify-between items-center">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                    {component.category}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {component.usageCount} uses
                  </span>
                </div>
                <div className="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                  <span>Used: {component.lastUsed.toLocaleDateString()}</span>
                  <span className={`capitalize ${component.status === 'deprecated' ? 'text-red-500' : component.status === 'experimental' ? 'text-yellow-500' : 'text-green-500'}`}>
                    {getStatusText(component.status)}
                  </span>
                </div>
              </div>
            ))}
            
            {filteredComponents.length === 0 && (
              <div className="col-span-full text-center py-12">
                <DatabaseIcon className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No components found</h3>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  {searchTerm || selectedCategory !== 'all'
                    ? "No components match your filters. Try adjusting your search."
                    : "No UI components available."}
                </p>
              </div>
            )}
          </div>
        </BubbleTab>

        <BubbleTab value="documentation" label="Documentation">
          {selectedComponent ? (
            <div className="space-y-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">{selectedComponent.name}</h2>
                    <p className="text-gray-600 dark:text-gray-400 mt-2">{selectedComponent.description}</p>
                  </div>
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    selectedComponent.status === 'active' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    selectedComponent.status === 'deprecated' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                    'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                  }`}>
                    {getStatusText(selectedComponent.status)}
                  </span>
                </div>
                
                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <h3 className="font-medium text-gray-900 dark:text-white mb-2">Category</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{selectedComponent.category}</p>
                  </div>
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <h3 className="font-medium text-gray-900 dark:text-white mb-2">Usage Count</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{selectedComponent.usageCount}</p>
                  </div>
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <h3 className="font-medium text-gray-900 dark:text-white mb-2">Last Used</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{selectedComponent.lastUsed.toLocaleString()}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Parameters</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Type</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Default</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Required</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Description</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {getComponentParameters(selectedComponent.id).map((param, index) => (
                        <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900 dark:text-white">{param.name}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900 dark:text-white">{param.type}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900 dark:text-white">{JSON.stringify(param.defaultValue)}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className={`text-sm ${param.required ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'}`}>
                              {param.required ? 'Yes' : 'No'}
                            </div>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                            {param.description}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Usage Example</h3>
                <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-sm text-gray-200">
                    {`import { ${selectedComponent.name} } from '@openevolve/bubblelab';\n\n` +
                     `<${selectedComponent.name}\n` +
                     getComponentParameters(selectedComponent.id).map(param => 
                       `  ${param.name}={${JSON.stringify(param.defaultValue)}}`
                     ).join('\n') +
                     `\n/>`}
                  </pre>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <FileText className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">No component selected</h3>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Select a component from the list to view its documentation
              </p>
            </div>
          )}
        </BubbleTab>

        <BubbleTab value="analytics" label="Analytics">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <BubbleCard className="p-5">
              <div className="flex items-center">
                <DatabaseIcon className="w-8 h-8 text-blue-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Components</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{components.length}</p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Components</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {components.filter(c => c.status === 'active').length}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <XCircle className="w-8 h-8 text-red-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Deprecated</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {components.filter(c => c.status === 'deprecated').length}
                  </p>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-5">
              <div className="flex items-center">
                <BarChart3 className="w-8 h-8 text-purple-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Usage</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {components.reduce((sum, c) => sum + c.usageCount, 0)}
                  </p>
                </div>
              </div>
            </BubbleCard>
          </div>

          {/* Usage by Category */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-8">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Usage by Category</h3>
            <div className="space-y-4">
              {categories.map(category => {
                const categoryComponents = components.filter(c => c.category === category);
                const totalUsage = categoryComponents.reduce((sum, c) => sum + c.usageCount, 0);
                
                return (
                  <div key={category} className="flex items-center">
                    <div className="w-32 text-sm font-medium text-gray-900 dark:text-white capitalize">
                      {category}
                    </div>
                    <div className="flex-1 ml-4">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">{categoryComponents.length} components</span>
                        <span className="font-medium text-gray-900 dark:text-white">{totalUsage} uses</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${(totalUsage / components.reduce((sum, c) => sum + c.usageCount, 0)) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Component Status Distribution */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Component Status Distribution</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                  <h4 className="font-medium text-gray-900 dark:text-white">Active</h4>
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                  {components.filter(c => c.status === 'active').length}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {Math.round((components.filter(c => c.status === 'active').length / components.length) * 100)}% of components
                </p>
              </div>
              
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
                  <h4 className="font-medium text-gray-900 dark:text-white">Experimental</h4>
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                  {components.filter(c => c.status === 'experimental').length}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {Math.round((components.filter(c => c.status === 'experimental').length / components.length) * 100)}% of components
                </p>
              </div>
              
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                  <h4 className="font-medium text-gray-900 dark:text-white">Deprecated</h4>
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                  {components.filter(c => c.status === 'deprecated').length}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {Math.round((components.filter(c => c.status === 'deprecated').length / components.length) * 100)}% of components
                </p>
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
          Export Components
        </BubbleButton>
      </div>
    </div>
  );
};

export default UIComponentsPage;