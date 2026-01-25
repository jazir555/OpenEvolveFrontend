import React, { memo, useState, useMemo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import {
  CogIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline';
import type { OpenEvolveNodeData } from '../types/nodeTypes';
import { BubbleButton, BubbleInput } from '../bubblelab';
import { withSafeComponent } from '@/components/shared/EnhancedErrorBoundary';
import { safeGet } from '@/utils/safeOperations';
import { HandleError } from '@/utils/errorHandlingDecorators';

/**
 * OpenEvolveNode - Base OpenEvolve node component
 *
 * This is the foundational node component that provides:
 * - Common UI structure for all OpenEvolve nodes
 * - Status indicators and state management
 * - Input/output handles
 * - Collapsible details panel
 * - Error and loading states
 * - Dark mode support with OpenEvolve purple/indigo theme
 */
const OpenEvolveNodeBase = memo((props: NodeProps<OpenEvolveNodeData>) => {
  const { data, selected } = props;
  const [isExpanded, setIsExpanded] = useState(false);
  const [showTooltip, setShowTooltip] = useState<string | null>(null);

  // Use safe property access for node data
  const nodeData = data || {};

  // Memoize status colors for performance with safe property access
  const statusConfig = useMemo(() => {
    const status = safeGet(nodeData, 'status', 'idle');

    switch (status) {
      case 'idle':
        return {
          bgColor: 'bg-neutral-800/90',
          borderColor: selected ? 'border-purple-500' : 'border-neutral-600',
          textColor: 'text-neutral-100',
          icon: <CogIcon className="w-4 h-4" />,
          iconBg: 'bg-neutral-700',
        };
      case 'running':
        return {
          bgColor: 'bg-purple-900/40',
          borderColor: 'border-purple-500',
          textColor: 'text-purple-100',
          icon: <ClockIcon className="w-4 h-4 animate-pulse" />,
          iconBg: 'bg-purple-600',
        };
      case 'completed':
        return {
          bgColor: 'bg-green-900/30',
          borderColor: 'border-green-500',
          textColor: 'text-green-100',
          icon: <CheckCircleIcon className="w-4 h-4" />,
          iconBg: 'bg-green-600',
        };
      case 'error':
        return {
          bgColor: 'bg-red-900/30',
          borderColor: 'border-red-500',
          textColor: 'text-red-100',
          icon: <ExclamationTriangleIcon className="w-4 h-4" />,
          iconBg: 'bg-red-600',
        };
      default:
        return {
          bgColor: 'bg-neutral-800/90',
          borderColor: 'border-neutral-600',
          textColor: 'text-neutral-100',
          icon: <CogIcon className="w-4 h-4" />,
          iconBg: 'bg-neutral-700',
        };
    }
  }, [safeGet(data, 'status', 'idle'), selected]);

  // Use safe property access for parameters and results
  const parameters = safeGet(nodeData, 'parameters', {});
  const hasParameters = parameters && Object.keys(parameters).length > 0;
  const results = safeGet(nodeData, 'results', {});
  const hasResults = results !== undefined && results !== null;
  const nodeStatus = safeGet(nodeData, 'status', 'idle');
  const hasErrors = nodeStatus === 'error' && results?.error;

  // Get node type icon with safe access
  const getNodeTypeIcon = () => {
    const nodeType = safeGet(nodeData, 'type', 'default');

    switch (nodeType) {
      case 'decomposition':
        return '🔍';
      case 'solution':
        return '💡';
      case 'verification':
        return '✅';
      default:
        return '⚙️';
    }
  };

  // Safe handler for parameter changes
  const handleParameterChange = (key: string, value: string) => {
    const onParameterChange = safeGet(nodeData, 'onParameterChange');
    if (typeof onParameterChange === 'function') {
      try {
        onParameterChange(key, value);
      } catch (error) {
        errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in onParameterChange' } });
      }
    }
  };

  // Safe handler for execution
  const handleExecute = (e: React.MouseEvent) => {
    e.stopPropagation();
    const onExecute = safeGet(nodeData, 'onExecute');
    if (typeof onExecute === 'function') {
      try {
        onExecute();
      } catch (error) {
        errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in onExecute' } });
      }
    }
  };

  return (
    <div
      className={`
        rounded-lg border-2 transition-all duration-300 cursor-pointer
        ${statusConfig.bgColor} ${statusConfig.borderColor}
        ${selected ? 'shadow-lg shadow-purple-500/20' : 'shadow-md'}
        hover:shadow-xl hover:shadow-purple-500/10
        min-w-[320px] max-w-[400px]
      `}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      {/* Input Handle (Left) */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="w-3 h-3 bg-purple-500 border-2 border-purple-300"
        style={{ left: -6 }}
      />

      {/* Output Handle (Right) */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="w-3 h-3 bg-purple-500 border-2 border-purple-300"
        style={{ right: -6 }}
      />

      {/* Header Section */}
      <div className="p-4 border-b border-neutral-700/50">
        {/* Top Row: Icon + Title + Status */}
        <div className="flex items-start gap-3">
          {/* Node Type Icon */}
          <div className={`
            flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center
            ${statusConfig.iconBg}
          `}>
            <span className="text-lg">{getNodeTypeIcon()}</span>
          </div>

          {/* Title and Description */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className={`text-sm font-semibold ${statusConfig.textColor} truncate`}>
                {safeGet(nodeData, 'displayName', 'Unnamed Node')}
              </h3>
              <div className="flex-shrink-0">
                {statusConfig.icon}
              </div>
            </div>
            {safeGet(nodeData, 'description') && (
              <p className="text-xs text-neutral-400 mt-1 truncate" title={safeGet(nodeData, 'description', '')}>
                {safeGet(nodeData, 'description', '')}
              </p>
            )}
          </div>

          {/* Info Icon with Tooltip */}
          {safeGet(nodeData, 'config') && (
            <div
              className="relative flex-shrink-0"
              onMouseEnter={() => setShowTooltip('info')}
              onMouseLeave={() => setShowTooltip(null)}
            >
              <InformationCircleIcon className="w-4 h-4 text-neutral-400 hover:text-purple-400 transition-colors" />
              {showTooltip === 'info' && (
                <div className="absolute top-full right-0 mt-2 px-3 py-2 text-xs bg-neutral-900 rounded-lg border border-neutral-700 shadow-xl z-50 w-48">
                  <div className="font-semibold text-purple-300 mb-1">Configuration</div>
                  <div className="text-neutral-300 space-y-1">
                    {Object.entries(safeGet(nodeData, 'config', {})).slice(0, 3).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-neutral-400">{key}:</span>
                        <span className="text-neutral-200 truncate ml-2">{String(value).substring(0, 15)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Expand/Collapse Button */}
        {hasParameters && (
          <BubbleButton
            variant="ghost"
            className="mt-3 w-full flex items-center justify-center gap-1 px-3 py-1.5 text-xs font-medium rounded-lg bg-neutral-700/50 text-neutral-300 hover:bg-neutral-700 transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
          >
            {isExpanded ? (
              <>
                <ChevronDownIcon className="w-3 h-3" />
                Hide Details
              </>
            ) : (
              <>
                <ChevronRightIcon className="w-3 h-3" />
                Show Details
              </>
            )}
          </BubbleButton>
        )}
      </div>

      {/* Expandable Details Section */}
      {(isExpanded || hasErrors) && (
        <div className="p-4 space-y-3 animate-in slide-in-from-top-2 duration-300">
          {/* Parameters Quick Edit */}
          {hasParameters && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-purple-300 uppercase tracking-wide">
                Parameters
              </div>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(parameters).slice(0, 4).map(([key, value]) => (
                  <div key={key} className="space-y-1">
                    <label className="block text-[10px] text-neutral-400 truncate" title={key}>
                      {key}
                    </label>
                    <BubbleInput
                      type="text"
                      value={String(value)}
                      onChange={(e) => {
                        e.stopPropagation();
                        handleParameterChange(key, e.target.value);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-full px-2 py-1 text-xs bg-neutral-700/50 border border-neutral-600 rounded text-neutral-200 focus:border-purple-500 focus:outline-none focus:ring-1 focus:ring-purple-500"
                    />
                  </div>
                ))}
              </div>
              {Object.keys(parameters).length > 4 && (
                <div className="text-[10px] text-neutral-500 text-center">
                  + {Object.keys(parameters).length - 4} more parameters
                </div>
              )}
            </div>
          )}

          {/* Results Display */}
          {hasResults && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-green-300 uppercase tracking-wide">
                Results
              </div>
              <div className="bg-neutral-900/50 rounded-lg p-3 border border-neutral-700">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {safeGet(results, 'score') !== undefined &&
                   Number.isFinite(Number(safeGet(results, 'score'))) && (
                    <div>
                      <span className="text-neutral-400">Score:</span>
                      <span className="ml-2 font-semibold text-green-400">
                        {Number(safeGet(results, 'score')).toFixed(2)}
                      </span>
                    </div>
                  )}
                  {safeGet(results, 'iterations') !== undefined && (
                    <div>
                      <span className="text-neutral-400">Iterations:</span>
                      <span className="ml-2 font-semibold text-purple-400">
                        {safeGet(results, 'iterations')}
                      </span>
                    </div>
                  )}
                  {safeGet(results, 'duration') !== undefined &&
                   Number.isFinite(Number(safeGet(results, 'duration'))) && (
                    <div>
                      <span className="text-neutral-400">Duration:</span>
                      <span className="ml-2 font-semibold text-blue-400">
                        {(Number(safeGet(results, 'duration')) / 1000).toFixed(1)}s
                      </span>
                    </div>
                  )}
                  {safeGet(results, 'confidence') !== undefined &&
                   Number.isFinite(Number(safeGet(results, 'confidence'))) && (
                    <div>
                      <span className="text-neutral-400">Confidence:</span>
                      <span className="ml-2 font-semibold text-indigo-400">
                        {(Number(safeGet(results, 'confidence')) * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {hasErrors && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-red-300 uppercase tracking-wide">
                Error
              </div>
              <div className="bg-red-950/30 rounded-lg p-3 border border-red-700/50">
                <p className="text-xs text-red-200 break-words">
                  {safeGet(results, 'error', 'Unknown error occurred')}
                </p>
              </div>
            </div>
          )}

          {/* Execution Button */}
          {safeGet(nodeData, 'onExecute') && nodeStatus !== 'running' && (
            <BubbleButton
              className="w-full px-4 py-2 text-sm font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:from-purple-700 hover:to-indigo-700 transition-all shadow-md hover:shadow-lg"
              onClick={handleExecute}
            >
              Execute {safeGet(nodeData, 'type', 'Node')}
            </BubbleButton>
          )}

          {/* Progress Indicator for Running State */}
          {nodeStatus === 'running' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-neutral-400">Progress</span>
                <span className="text-purple-300 font-semibold">
                  {Number.isFinite(Number(safeGet(nodeData, 'progress', 0)))
                    ? `${Number(safeGet(nodeData, 'progress', 0)).toFixed(0)}%`
                    : 'Running...'}
                </span>
              </div>
              <div className="w-full bg-neutral-700 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-purple-500 to-indigo-500 h-2 transition-all duration-300 ease-out"
                  style={{
                    width: `${safeGet(nodeData, 'progress', 50)}%`
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

OpenEvolveNodeBase.displayName = 'OpenEvolveNode';

export const OpenEvolveNode = withSafeComponent(OpenEvolveNodeBase, 'OpenEvolveNode');

export default OpenEvolveNode;
