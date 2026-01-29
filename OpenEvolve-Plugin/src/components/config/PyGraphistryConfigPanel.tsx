/**
 * PyGraphistry Configuration Panel
 * 
 * Configuration panel for PyGraphistry node settings
 */

import React, { useState, useEffect } from 'react';
import { PyGraphistryNodeConfig } from '../../nodes/PyGraphistryNode';

interface PyGraphistryConfigPanelProps {
  config: PyGraphistryNodeConfig;
  onUpdate: (config: PyGraphistryNodeConfig) => void;
}

const PyGraphistryConfigPanel: React.FC<PyGraphistryConfigPanelProps> = ({ config, onUpdate }) => {
  const [localConfig, setLocalConfig] = useState<PyGraphistryNodeConfig>({
    layout: 'force_directed',
    clustering: false,
    clusteringMethod: 'dbscan',
    enableGPUAcceleration: true,
    serverUrl: 'http://localhost:8000',
    enableBackendExecution: true,
    backendUrl: 'http://localhost:8000',
    ...config
  });

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      setLocalConfig(prev => ({
        ...prev,
        ...config
      }));
    } catch (effectError) {
      setError('Error loading configuration');
      errorLogger.logError(effectError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'PyGraphistryConfigPanel effect error' } });
    }
  }, [config]);

  const handleChange = (field: keyof PyGraphistryNodeConfig, value: any) => {
    try {
      const updatedConfig = { ...localConfig, [field]: value };
      setLocalConfig(updatedConfig);
      onUpdate(updatedConfig);
    } catch (updateError) {
      setError('Error updating configuration');
      errorLogger.logError(updateError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Configuration update error' } });
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">PyGraphistry Configuration</h3>
      
      {/* Error display */}
      {error && (
        <div className="p-3 bg-red-100 text-red-700 text-sm rounded-md">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Layout
          </label>
          <select
            value={localConfig.layout}
            onChange={(e) => {
              try {
                handleChange('layout', e.target.value as any);
              } catch (inputError) {
                setError('Error updating layout');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Layout input error' } });
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          >
            <option value="force_directed">Force Directed</option>
            <option value="circular">Circular</option>
            <option value="hierarchical">Hierarchical</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Backend URL
          </label>
          <input
            type="text"
            value={localConfig.backendUrl}
            onChange={(e) => {
              try {
                handleChange('backendUrl', e.target.value);
              } catch (inputError) {
                setError('Error updating backend URL');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend URL input error' } });
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="flex items-center">
          <input
            type="checkbox"
            id="clustering"
            checked={localConfig.clustering}
            onChange={(e) => {
              try {
                handleChange('clustering', e.target.checked);
              } catch (inputError) {
                setError('Error updating clustering setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Clustering checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="clustering" className="ml-2 block text-sm text-gray-900">
            Clustering
          </label>
        </div>
        
        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableGPUAcceleration"
            checked={localConfig.enableGPUAcceleration}
            onChange={(e) => {
              try {
                handleChange('enableGPUAcceleration', e.target.checked);
              } catch (inputError) {
                setError('Error updating GPU acceleration setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'GPU acceleration checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableGPUAcceleration" className="ml-2 block text-sm text-gray-900">
            GPU Acceleration
          </label>
        </div>
        
        <div className="flex items-center">
          <input
            type="checkbox"
            id="enableBackendExecution"
            checked={localConfig.enableBackendExecution}
            onChange={(e) => {
              try {
                handleChange('enableBackendExecution', e.target.checked);
              } catch (inputError) {
                setError('Error updating backend execution setting');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend execution checkbox error' } });
              }
            }}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="enableBackendExecution" className="ml-2 block text-sm text-gray-900">
            Backend Execution
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Clustering Method
          </label>
          <select
            value={localConfig.clusteringMethod}
            onChange={(e) => {
              try {
                handleChange('clusteringMethod', e.target.value as any);
              } catch (inputError) {
                setError('Error updating clustering method');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Clustering method input error' } });
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          >
            <option value="dbscan">DBSCAN</option>
            <option value="kmeans">K-Means</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Server URL
          </label>
          <input
            type="text"
            value={localConfig.serverUrl}
            onChange={(e) => {
              try {
                handleChange('serverUrl', e.target.value);
              } catch (inputError) {
                setError('Error updating server URL');
                errorLogger.logError(inputError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Server URL input error' } });
              }
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          />
        </div>
      </div>
    </div>
  );
};

export default PyGraphistryConfigPanel;