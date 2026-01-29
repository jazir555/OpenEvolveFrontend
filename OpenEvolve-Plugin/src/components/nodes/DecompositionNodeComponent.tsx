// @ts-nocheck
import React, { memo, useState, useMemo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import {
  MagnifyingGlassIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ChartBarIcon,
  DocumentTextIcon,
  LinkIcon,
  SparklesIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import type { DecompositionNodeData, SubProblem, DependencyInfo } from '../../types';
import { BubbleButton } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

/**
 * DecompositionNodeComponent - Specialized node for problem decomposition
 *
 * Features:
 * - Visual representation of decomposed sub-problems
 * - Expandable sub-problem list with status indicators
 * - Dependency graph preview
 * - Quality metrics dashboard
 * - Progress tracking
 * - Interactive parameter editing
 */
const DecompositionNodeComponentBase = memo((props: NodeProps<DecompositionNodeData>) => {
  const { data, selected } = props;
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedSubProblems, setExpandedSubProblems] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<'overview' | 'subproblems' | 'dependencies'>('overview');

  // Type assertion to access extended properties
  const nodeData = data as DecompositionNodeData;

  // Memoize sub-problems stats
  const subProblemStats = useMemo(() => {
    if (!nodeData.subProblems) return null;
    const total = nodeData.subProblems.length;
    if (total === 0) {
      return {
        total: 0,
        completed: 0,
        inProgress: 0,
        blocked: 0,
        avgComplexity: 0,
      };
    }

    return {
      total,
      completed: nodeData.subProblems.filter(sp => sp.status === 'completed').length,
      inProgress: nodeData.subProblems.filter(sp => sp.status === 'in_progress').length,
      blocked: nodeData.subProblems.filter(sp => sp.status === 'blocked').length,
      avgComplexity: nodeData.subProblems.reduce((sum, sp) => sum + (sp.complexity ?? 0), 0) / total,
    };
  }, [nodeData.subProblems]);

  // Quality score color
  const getQualityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getQualityBg = (score: number) => {
    if (score >= 0.8) return 'bg-green-500/20';
    if (score >= 0.6) return 'bg-yellow-500/20';
    return 'bg-red-500/20';
  };

  // Status badge component
  const StatusBadge = ({ status }: { status: SubProblem['status'] }) => {
    const config = {
      pending: { bg: 'bg-neutral-700/50', text: 'text-neutral-300', label: 'Pending' },
      in_progress: { bg: 'bg-blue-500/20', text: 'text-blue-300', label: 'In Progress' },
      completed: { bg: 'bg-green-500/20', text: 'text-green-300', label: 'Completed' },
      blocked: { bg: 'bg-red-500/20', text: 'text-red-300', label: 'Blocked' },
    };
    const { bg, text, label } = config[status] || config.pending;
    return (
      <span className={`px-2 py-0.5 text-[10px] font-medium rounded-full ${bg} ${text}`}>
        {label}
      </span>
    );
  };

  // Toggle sub-problem expansion
  const toggleSubProblem = (id: string) => {
    setExpandedSubProblems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  return (
    <div
      className={`
        rounded-lg border-2 transition-all duration-300 cursor-pointer
        bg-indigo-950/50 ${selected ? 'border-indigo-500 shadow-lg shadow-indigo-500/20' : 'border-indigo-700 shadow-md'}
        hover:shadow-xl hover:shadow-indigo-500/10
        min-w-[360px] max-w-[440px]
      `}
    >
      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="w-3 h-3 bg-indigo-500 border-2 border-indigo-300"
        style={{ left: -6 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="w-3 h-3 bg-indigo-500 border-2 border-indigo-300"
        style={{ right: -6 }}
      />

      {/* Header */}
      <div className="p-4 border-b border-indigo-800/50">
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg">
            <MagnifyingGlassIcon className="w-5 h-5 text-white" />
          </div>

          {/* Title */}
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-indigo-100 truncate flex items-center gap-2">
              {nodeData.displayName as any}
              {nodeData.status === 'running' && (
                <SparklesIcon className="w-4 h-4 animate-pulse text-indigo-400" />
              )}
            </h3>
            {nodeData.description && (
              <p className="text-xs text-indigo-300/70 mt-1 truncate">{nodeData.description as any}</p>
            )}
          </div>
        </div>

        {/* Quality Score Badge */}
        {nodeData.qualityScore !== undefined && Number.isFinite(Number(nodeData.qualityScore)) && (
          <div className={`
            mt-3 px-3 py-2 rounded-lg ${getQualityBg(nodeData.qualityScore)} border border-indigo-600/30
          `}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-indigo-200">Quality Score</span>
              <span className={`text-lg font-bold ${getQualityColor(nodeData.qualityScore)}`}>
                {(Number(nodeData.qualityScore) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-indigo-800/50">
        {[
          { id: 'overview' as const, label: 'Overview', icon: ChartBarIcon },
          { id: 'subproblems' as const, label: 'Sub-Problems', icon: DocumentTextIcon, count: subProblemStats?.total },
          { id: 'dependencies' as const, label: 'Dependencies', icon: LinkIcon },
        ].map(tab => (
          <BubbleButton
            key={tab.id}
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab(tab.id);
            }}
            variant="ghost"
            className={`
              flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors
              ${activeTab === tab.id
                ? 'bg-indigo-600/30 text-indigo-200 border-b-2 border-indigo-400'
                : 'text-indigo-400 hover:bg-indigo-900/30 hover:text-indigo-300'
              }
            `}
          >
            <tab.icon className="w-3.5 h-3.5" />
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span className={`
                px-1.5 py-0.5 rounded-full text-[10px]
                ${activeTab === tab.id ? 'bg-indigo-500/40' : 'bg-indigo-800/40'}
              `}>
                {tab.count}
              </span>
            )}
          </BubbleButton>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-4">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-3 animate-in fade-in duration-200">
            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-2">
              {nodeData.qualityScore !== undefined && Number.isFinite(Number(nodeData.qualityScore)) && (
                <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30">
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Quality</div>
                  <div className={`text-lg font-bold ${getQualityColor(nodeData.qualityScore)}`}>
                    {(Number(nodeData.qualityScore) * 100).toFixed(0)}%
                  </div>
                </div>
              )}
              {nodeData.complexity !== undefined && Number.isFinite(Number(nodeData.complexity)) && (
                <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30">
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Complexity</div>
                  <div className="text-lg font-bold text-purple-400">
                    {Number(nodeData.complexity).toFixed(1)}
                  </div>
                </div>
              )}
              {nodeData.completeness !== undefined && Number.isFinite(Number(nodeData.completeness)) && (
                <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30">
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Completeness</div>
                  <div className="text-lg font-bold text-blue-400">
                    {(Number(nodeData.completeness) * 100).toFixed(0)}%
                  </div>
                </div>
              )}
              {subProblemStats && (
                <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30">
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Sub-Problems</div>
                  <div className="text-lg font-bold text-green-400">
                    {subProblemStats.completed}/{subProblemStats.total}
                  </div>
                </div>
              )}
            </div>

            {/* Progress Bar */}
            {subProblemStats && subProblemStats.total > 0 && (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs text-indigo-300">
                  <span>Decomposition Progress</span>
                  <span className="font-semibold">
                    {((subProblemStats.completed / subProblemStats.total) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 transition-all duration-300"
                    style={{ width: `${(subProblemStats.completed / subProblemStats.total) * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* Execution Button */}
            {nodeData.onExecute && nodeData.status !== 'running' && (
              <BubbleButton
                className="w-full px-4 py-2 text-sm font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 transition-all shadow-md"
                onClick={(e) => {
                  e.stopPropagation();
                  nodeData.onExecute?.();
                }}
              >
                Execute Decomposition
              </BubbleButton>
            )}
          </div>
        )}

        {/* Sub-Problems Tab */}
        {activeTab === 'subproblems' && (
          <div className="space-y-2 animate-in fade-in duration-200">
            {nodeData.subProblems && nodeData.subProblems.length > 0 ? (
              <>
                {nodeData.subProblems.map((sp) => (
                  <div
                    key={sp.id}
                    className="bg-neutral-900/50 rounded-lg border border-indigo-800/30 overflow-hidden"
                  >
                    {/* Sub-problem header */}
                    <div
                      className="p-3 cursor-pointer hover:bg-indigo-900/20 transition-colors"
                      onClick={() => toggleSubProblem(sp.id)}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            {expandedSubProblems.has(sp.id) ? (
                              <ChevronDownIcon className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                            ) : (
                              <ChevronRightIcon className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                            )}
                            <h4 className="text-sm font-medium text-indigo-100 truncate">
                              {sp.title}
                            </h4>
                          </div>
                          <StatusBadge status={sp.status} />
                        </div>
                        <div className="text-[10px] text-purple-400 flex-shrink-0">
                          C: {Number.isFinite(Number(sp.complexity))
                            ? Number(sp.complexity).toFixed(1)
                            : 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* Expanded details */}
                    {expandedSubProblems.has(sp.id) && (
                      <div className="px-3 pb-3 animate-in slide-in-from-top-1 duration-200">
                        <p className="text-xs text-indigo-300/80 mb-2">{sp.description}</p>
                        {sp.dependencies?.length > 0 && (
                          <div className="text-[10px] text-neutral-400">
                            <span className="font-semibold">Depends on:</span> {sp.dependencies.join(', ')}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </>
            ) : (
              <div className="text-center py-6 text-neutral-500 text-xs">
                No sub-problems yet. Execute decomposition to generate.
              </div>
            )}
          </div>
        )}

        {/* Dependencies Tab */}
        {activeTab === 'dependencies' && (
          <div className="space-y-3 animate-in fade-in duration-200">
            {nodeData.dependencyGraph ? (
              <>
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30 text-center">
                    <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Total</div>
                    <div className="text-lg font-bold text-indigo-200">
                      {nodeData.dependencyGraph.totalDependencies ?? 0}
                    </div>
                  </div>
                  <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30 text-center">
                    <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Critical</div>
                    <div className="text-lg font-bold text-yellow-400">
                      {nodeData.dependencyGraph.criticalPath ?? 0}
                    </div>
                  </div>
                  <div className="bg-neutral-900/50 rounded-lg p-3 border border-indigo-800/30 text-center">
                    <div className="text-[10px] text-indigo-400 uppercase tracking-wide mb-1">Circular</div>
                    <div className={`text-lg font-bold ${(nodeData.dependencyGraph.circularDeps ?? 0) > 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {nodeData.dependencyGraph.circularDeps ?? 0}
                    </div>
                  </div>
                </div>

                {(nodeData.dependencyGraph.circularDeps ?? 0) > 0 && (
                  <div className="flex items-start gap-2 p-3 bg-yellow-950/30 rounded-lg border border-yellow-700/30">
                    <ExclamationTriangleIcon className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-yellow-200">
                      Circular dependencies detected. This may affect execution order.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-6 text-neutral-500 text-xs">
                No dependency data available. Execute decomposition to analyze.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

DecompositionNodeComponentBase.displayName = 'DecompositionNodeComponent';

export const DecompositionNodeComponent = withComponentBoundary(
  DecompositionNodeComponentBase,
  'DecompositionNodeComponent'
);

export default DecompositionNodeComponent;
