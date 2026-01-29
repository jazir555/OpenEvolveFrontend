/**
 * Research Quest Node Component
 *
 * React component for the Research Quest node in the OpenEvolve workflow
 */

import React, { useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { ResearchQuestNode } from '../../nodes/ResearchQuestNode';

interface ResearchQuestNodeData {
  label?: string;
  config?: any;
  stage?: string;
  taskDescription?: string;
  maxHypotheses?: number;
  enableMultiLayer?: boolean;
  enableCausalInference?: boolean;
  enableTemporalAnalysis?: boolean;
  enableBiasAssessment?: boolean;
  enableFalsificationChecks?: boolean;
  enableImpactScoring?: boolean;
  enableInterdisciplinaryBridges?: boolean;
  enableKnowledgeGapDetection?: boolean;
  enableProbabilisticConfidence?: boolean;
  enableGraphRestructuring?: boolean;
  enableTopologyAnalysis?: boolean;
  enableInformationTheoryMetrics?: boolean;
  enableAttributionTracking?: boolean;
  enableStatisticalPowerAnalysis?: boolean;
  enableMultiScaleAnalysis?: boolean;
  enableCostEstimation?: boolean;
  enableSelfAudit?: boolean;
  enableEvidenceIntegration?: boolean;
  enablePruningMerging?: boolean;
  enableSubgraphExtraction?: boolean;
  enableReflection?: boolean;
  enableComposition?: boolean;
  enableHypothesisGeneration?: boolean;
  enableTaskDecomposition?: boolean;
  enableInitialization?: boolean;
  enableBackendExecution?: boolean;
  backendUrl?: string;
}

type ResearchQuestNodeComponentProps = NodeProps<ResearchQuestNodeData>;

const ResearchQuestNodeComponent: React.FC<ResearchQuestNodeComponentProps> = ({ data, isConnectable }) => {
  const [expanded, setExpanded] = useState(false);
  const [stage, setStage] = useState(data.stage || 'initialization');
  const [taskDescription, setTaskDescription] = useState(data.taskDescription || '');
  const [maxHypotheses, setMaxHypotheses] = useState<number>(data.maxHypotheses || 5);
  const [enableMultiLayer, setEnableMultiLayer] = useState<boolean>(data.enableMultiLayer ?? true);
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
      if (maxHypotheses < 3 || maxHypotheses > 5) {
        setError('Max hypotheses must be between 3 and 5');
      } else {
        setError(null);
      }
    } catch (effectError) {
      setError('Error initializing component');
      errorLogger.logError(effectError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'ResearchQuestNodeComponent effect error' } });
    }
  }, [maxHypotheses]);

  return (
    <div className={`rounded-lg border-2 bg-white shadow-lg min-w-[280px] ${
      expanded ? 'p-4 border-blue-500' : 'p-2 border-gray-300'
    }`}>
      {/* Node header */}
      <div
        className="flex items-center cursor-pointer"
        onClick={toggleExpanded}
      >
        <div className="mr-2 text-purple-600">
          🔬 {/* Research icon */}
        </div>
        <div className="font-bold text-sm truncate flex-grow">
          {data.label || 'Research Quest'}
        </div>
        <div className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">
          {stage}
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
          {/* Stage selector */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">Stage</label>
            <select
              value={stage}
              onChange={(e) => {
                try {
                  setStage(e.target.value);
                } catch (changeError) {
                  setError('Error changing stage');
                  errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Stage change error' } });
                }
              }}
              className="w-full text-xs border border-gray-300 rounded px-2 py-1"
            >
              <option value="initialization">Initialization</option>
              <option value="decomposition">Decomposition</option>
              <option value="hypothesis_planning">Hypothesis Planning</option>
              <option value="evidence_integration">Evidence Integration</option>
              <option value="pruning_merging">Pruning/Merging</option>
              <option value="subgraph_extraction">Subgraph Extraction</option>
              <option value="composition">Composition</option>
              <option value="reflection">Reflection</option>
              <option value="get_graph_summary">Get Graph Summary</option>
              <option value="export_graph_data">Export Graph Data</option>
            </select>
          </div>

          {/* Task description */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">Task Description</label>
            <textarea
              value={taskDescription}
              onChange={(e) => {
                try {
                  setTaskDescription(e.target.value);
                } catch (changeError) {
                  setError('Error updating task description');
                  errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Task description change error' } });
                }
              }}
              className="w-full text-xs border border-gray-300 rounded px-2 py-1 h-16"
              placeholder="Describe the research task..."
            />
          </div>

          {/* Configuration options */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">Max Hypotheses</label>
              <input
                type="number"
                min="3"
                max="5"
                value={maxHypotheses}
                onChange={(e) => {
                  try {
                    const value = parseInt(e.target.value);
                    if (!isNaN(value)) {
                      setMaxHypotheses(value);
                    }
                  } catch (changeError) {
                    setError('Error updating max hypotheses');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Max hypotheses change error' } });
                  }
                }}
                className="w-full text-xs border border-gray-300 rounded px-2 py-1"
              />
            </div>

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
          </div>

          {/* Boolean configuration options */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="enableMultiLayer"
                checked={enableMultiLayer}
                onChange={(e) => {
                  try {
                    setEnableMultiLayer(e.target.checked);
                  } catch (changeError) {
                    setError('Error updating multi-layer setting');
                    errorLogger.logError(changeError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Multi-layer change error' } });
                  }
                }}
                className="mr-1"
              />
              <label htmlFor="enableMultiLayer" className="text-xs text-gray-700">Multi-Layer</label>
            </div>

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
            <span className="text-gray-500">Research Quest Node</span>
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

export default ResearchQuestNodeComponent;