/**
 * PyGraphistry Node Component
 * 
 * React component for the PyGraphistry node in the OpenEvolve workflow
 */

import React, { useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { PyGraphistryNode } from '../../nodes/PyGraphistryNode';

interface PyGraphistryNodeData {
  label?: string;
  config?: any;
  layout?: string;
  clustering?: boolean;
  clusteringMethod?: string;
  enableGPUAcceleration?: boolean;
  serverUrl?: string;
  enableBackendExecution?: boolean;
  backendUrl?: string;
}

type PyGraphistryNodeComponentProps = NodeProps<PyGraphistryNodeData>;

const PyGraphistryNodeComponent: React.FC<PyGraphistryNodeComponentProps> = ({ data, isConnectable }) => {
  const [expanded, setExpanded] = useState(false);
  const [layout, setLayout] = useState(data.layout || 'force_directed');
  const [clustering, setClustering] = useState<boolean>(data.clustering ?? false);
  const [clusteringMethod, setClusteringMethod] = useState(data.clusteringMethod || 'dbscan');
  const [enableGPUAcceleration, setEnableGPUAcceleration] = useState<boolean>(data.enableGPUAcceleration ?? true);
  const [enableBackendExecution, setEnableBackendExecution] = useState<boolean>(data.enableBackendExecution ?? true);
  const [backendUrl, setBackendUrl] = useState(data.backendUrl || 'http://localhost:8000');
  const [error, setError] = useState<string | null>(null);

  // Toggle expanded view
  const toggleExpanded = () => {
    setExpanded(!expanded);
  };

  // Handle errors gracefully
  useEffect(() => {
    try {
      // Validate inputs on mount/update
      if (!['force_directed', 'circular', 'hierarchical'].includes(layout)) {
        setError('Invalid layout selected');
      } else {
        setError(null);
      }
    } catch (effectError) {
      setError('Error initializing component');
      errorLogger.logError(effectError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'PyGraphistryNodeComponent effect error' } });
    }
  }, [layout]);

  return (
    <div className={`rounded-lg border-2 bg-white shadow-lg min-w-[280px] ${
      expanded ? 'p-4 border-blue-500' : 'p-2 border-gray-300'
    }`}>
      {/* Node header */}
      <div 
        className="flex items-center cursor-pointer"
        onClick={toggleExpanded}
      >
        <div className="mr-2 text-blue-600">
          📊 {/* Graph icon */}
        </div>
        <div className="font-bold text-sm truncate flex-grow">
          {data.label || 'PyGraphistry Visualization'}
        </div>
        <div className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
          {layout}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mt-2 p-2 bg-red-100 text-red-700 text-xs rounded">
          {error}
        </div>
      )}

      {/* Expandable content */}
      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          {/* Layout selector */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">Layout</label>
            <select
              value={layout}
              onChange={(e) => {
                try {
                  setLayout(e.target.value);
                } catch (changeError) {
                  setError('Error changing layout');
                  errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Layout change error' } });
                }
              }}
              className="w-full text-xs border border-gray-300 rounded px-2 py-1"
            >
              <option value="force_directed">Force Directed</option>
              <option value="circular">Circular</option>
              <option value="hierarchical">Hierarchical</option>
            </select>
          </div>

          {/* Configuration options */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">Backend URL</label>
              <input
                type="text"
                value={backendUrl}
                onChange={(e) => {
                  try {
                    setBackendUrl(e.target.value);
                  } catch (changeError) {
                    setError('Error updating backend URL');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend URL change error' } });
                  }
                }}
                className="w-full text-xs border border-gray-300 rounded px-2 py-1"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">Clustering Method</label>
              <select
                value={clusteringMethod}
                onChange={(e) => {
                  try {
                    setClusteringMethod(e.target.value);
                  } catch (changeError) {
                    setError('Error changing clustering method');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Clustering method change error' } });
                  }
                }}
                className="w-full text-xs border border-gray-300 rounded px-2 py-1"
              >
                <option value="dbscan">DBSCAN</option>
                <option value="kmeans">K-Means</option>
              </select>
            </div>
          </div>

          {/* Boolean configuration options */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="clustering"
                checked={clustering}
                onChange={(e) => {
                  try {
                    setClustering(e.target.checked);
                  } catch (changeError) {
                    setError('Error updating clustering setting');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Clustering change error' } });
                  }
                }}
                className="mr-1"
              />
              <label htmlFor="clustering" className="text-xs text-gray-700">Clustering</label>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="enableGPUAcceleration"
                checked={enableGPUAcceleration}
                onChange={(e) => {
                  try {
                    setEnableGPUAcceleration(e.target.checked);
                  } catch (changeError) {
                    setError('Error updating GPU acceleration setting');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'GPU acceleration change error' } });
                  }
                }}
                className="mr-1"
              />
              <label htmlFor="enableGPUAcceleration" className="text-xs text-gray-700">GPU Accel</label>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="enableBackendExecution"
                checked={enableBackendExecution}
                onChange={(e) => {
                  try {
                    setEnableBackendExecution(e.target.checked);
                  } catch (changeError) {
                    setError('Error updating backend execution setting');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Backend execution change error' } });
                  }
                }}
                className="mr-1"
              />
              <label htmlFor="enableBackendExecution" className="text-xs text-gray-700">Backend</label>
            </div>
          </div>

          {/* Status indicator */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-gray-500">PyGraphistry Node</span>
            <span className={`px-2 py-1 rounded text-xs ${
              enableBackendExecution ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
            }`}>
              {enableBackendExecution ? 'Backend' : 'Local'}
            </span>
          </div>
        </div>
      )}

      {/* Node handles */}
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
        className="w-3 h-3 bg-gray-500"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
        className="w-3 h-3 bg-gray-500"
      />
    </div>
  );
};

export default PyGraphistryNodeComponent;