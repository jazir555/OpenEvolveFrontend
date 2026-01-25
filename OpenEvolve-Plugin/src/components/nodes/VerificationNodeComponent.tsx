// @ts-nocheck
import React, { memo, useState, useMemo, useEffect } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import {
  CheckCircleIcon,
  XCircleIcon,
  ShieldCheckIcon,
  ClipboardDocumentCheckIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline';
import type { VerificationNodeData, VerificationQualityMetrics, VerificationRequirement } from '../../types';
import { BubbleButton } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

/**
 * VerificationNodeComponent - Specialized node for solution verification
 *
 * Features:
 * - Pass/Fail badge with visual indicator
 * - Quality metrics dashboard with gauges
 * - Requirement checklist with categories
 * - Verification score display
 * - Detailed check results
 * - Expandable requirement details
 * - Color-coded status indicators
 */
const VerificationNodeComponentBase = memo((props: NodeProps) => {
  const { data, selected } = props;

  // Type assertion to access extended properties
  const nodeData = data as VerificationNodeData;

  const [isExpanded, setIsExpanded] = useState(true);
  const [expandedRequirements, setExpandedRequirements] = useState<Set<string>>(new Set());
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // Get unique categories
  const categories = useMemo(() => {
    if (!nodeData.requirements) return ['all'];
    const cats = [
      'all',
      ...new Set(
        nodeData.requirements.map((r) => r.category || 'uncategorized')
      ),
    ];
    return cats;
  }, [nodeData.requirements]);

  useEffect(() => {
    if (!categories.includes(selectedCategory)) {
      setSelectedCategory('all');
    }
  }, [categories, selectedCategory]);

  // Filter requirements by category
  const filteredRequirements = useMemo(() => {
    if (!nodeData.requirements) return [];
    if (selectedCategory === 'all') return nodeData.requirements;
    return nodeData.requirements.filter(
      (r) => (r.category || 'uncategorized') === selectedCategory
    );
  }, [nodeData.requirements, selectedCategory]);

  // Calculate stats
  const stats = useMemo(() => {
    if (!nodeData.requirements) return null;
    return {
      total: nodeData.requirements.length,
      pass: nodeData.requirements.filter(r => r.status === 'pass').length,
      fail: nodeData.requirements.filter(r => r.status === 'fail').length,
      warning: nodeData.requirements.filter(r => r.status === 'warning').length,
      skipped: nodeData.requirements.filter(r => r.status === 'skipped').length,
    };
  }, [nodeData.requirements]);

  // Status config
  const statusConfig = useMemo(() => {
    switch (nodeData.verificationStatus) {
      case 'pass':
        return {
          bgColor: 'bg-green-950/50',
          borderColor: selected ? 'border-green-500' : 'border-green-700',
          icon: <CheckCircleIcon className="w-6 h-6" />,
          iconBg: 'bg-green-600',
          textColor: 'text-green-100',
          badgeText: 'PASS',
          badgeBg: 'bg-green-600/30',
          badgeBorder: 'border-green-500',
        };
      case 'fail':
        return {
          bgColor: 'bg-red-950/50',
          borderColor: selected ? 'border-red-500' : 'border-red-700',
          icon: <XCircleIcon className="w-6 h-6" />,
          iconBg: 'bg-red-600',
          textColor: 'text-red-100',
          badgeText: 'FAIL',
          badgeBg: 'bg-red-600/30',
          badgeBorder: 'border-red-500',
        };
      case 'warning':
        return {
          bgColor: 'bg-yellow-950/50',
          borderColor: selected ? 'border-yellow-500' : 'border-yellow-700',
          icon: <ExclamationTriangleIcon className="w-6 h-6" />,
          iconBg: 'bg-yellow-600',
          textColor: 'text-yellow-100',
          badgeText: 'WARNING',
          badgeBg: 'bg-yellow-600/30',
          badgeBorder: 'border-yellow-500',
        };
      default:
        return {
          bgColor: 'bg-neutral-900/50',
          borderColor: selected ? 'border-neutral-500' : 'border-neutral-700',
          icon: <ClipboardDocumentCheckIcon className="w-6 h-6" />,
          iconBg: 'bg-neutral-600',
          textColor: 'text-neutral-100',
          badgeText: 'PENDING',
          badgeBg: 'bg-neutral-600/30',
          badgeBorder: 'border-neutral-500',
        };
    }
  }, [nodeData.verificationStatus, selected]);

  // Requirement status icon
  const RequirementStatusIcon = ({ status }: { status: VerificationRequirement['status'] }) => {
    switch (status) {
      case 'pass':
        return <CheckCircleIcon className="w-4 h-4 text-green-400 flex-shrink-0" />;
      case 'fail':
        return <XCircleIcon className="w-4 h-4 text-red-400 flex-shrink-0" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-4 h-4 text-yellow-400 flex-shrink-0" />;
      case 'skipped':
        return <InformationCircleIcon className="w-4 h-4 text-neutral-500 flex-shrink-0" />;
    }
  };

  // Quality metric bar component
  const MetricBar = ({ label, value, color }: { label: string; value: number; color: string }) => {
    const safeValue = Number.isFinite(value) ? value : 0;
    const percentage = Math.min(Math.max(safeValue * 100, 0), 100);
    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-neutral-400">{label}</span>
          <span className={`font-semibold ${color}`}>{percentage.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
          <div
            className={`h-2 transition-all duration-300 ${color.replace('text-', 'bg-')}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  // Toggle requirement expansion
  const toggleRequirement = (id: string) => {
    setExpandedRequirements(prev => {
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
        ${statusConfig.bgColor} ${statusConfig.borderColor}
        ${selected ? 'shadow-lg' : 'shadow-md'}
        hover:shadow-xl
        min-w-[400px] max-w-[520px]
      `}
    >
      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="w-3 h-3 bg-green-500 border-2 border-green-300"
        style={{ left: -6 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="w-3 h-3 bg-green-500 border-2 border-green-300"
        style={{ right: -6 }}
      />

      {/* Header */}
      <div className="p-4 border-b border-neutral-700/50">
        <div className="flex items-start gap-3">
          {/* Status Icon */}
          <div className={`
            flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center shadow-lg
            ${statusConfig.iconBg}
          `}>
            <div className="text-white">
              {statusConfig.icon}
            </div>
          </div>

          {/* Title and Badge */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className={`text-sm font-semibold ${statusConfig.textColor} truncate`}>
                {nodeData.displayName as any}
              </h3>
              <span className={`
                px-2 py-0.5 text-[10px] font-bold rounded-full border
                ${statusConfig.badgeBg} ${statusConfig.badgeBorder} ${statusConfig.textColor}
              `}>
                {statusConfig.badgeText}
              </span>
            </div>
            {nodeData.description && (
              <p className="text-xs text-neutral-400 mt-1">{nodeData.description as any}</p>
            )}
          </div>

          {/* Verification Score */}
          {nodeData.verificationScore !== undefined && Number.isFinite(Number(nodeData.verificationScore)) && (
            <div className="flex-shrink-0 text-center">
              <div className={`text-2xl font-bold ${
                nodeData.verificationScore >= 0.8 ? 'text-green-400' :
                nodeData.verificationScore >= 0.6 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {(Number(nodeData.verificationScore) * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-neutral-500">Score</div>
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Quick Stats */}
        {stats && (
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-neutral-900/50 rounded-lg p-3 border border-neutral-700/50 text-center">
              <div className="text-lg font-bold text-neutral-200">{stats.total}</div>
              <div className="text-[10px] text-neutral-500 uppercase">Total</div>
            </div>
            <div className="bg-green-950/30 rounded-lg p-3 border border-green-700/30 text-center">
              <div className="text-lg font-bold text-green-400">{stats.pass}</div>
              <div className="text-[10px] text-green-500/70 uppercase">Pass</div>
            </div>
            <div className="bg-red-950/30 rounded-lg p-3 border border-red-700/30 text-center">
              <div className="text-lg font-bold text-red-400">{stats.fail}</div>
              <div className="text-[10px] text-red-500/70 uppercase">Fail</div>
            </div>
            <div className="bg-yellow-950/30 rounded-lg p-3 border border-yellow-700/30 text-center">
              <div className="text-lg font-bold text-yellow-400">{stats.warning}</div>
              <div className="text-[10px] text-yellow-500/70 uppercase">Warning</div>
            </div>
          </div>
        )}

        {/* Quality Metrics */}
        {nodeData.qualityMetrics && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <ShieldCheckIcon className="w-4 h-4 text-neutral-400" />
              <span className="text-xs font-semibold text-neutral-300 uppercase tracking-wide">
                Quality Metrics
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <MetricBar label="Accuracy" value={nodeData.qualityMetrics.accuracy} color="text-green-400" />
              <MetricBar label="Completeness" value={nodeData.qualityMetrics.completeness} color="text-blue-400" />
              <MetricBar label="Consistency" value={nodeData.qualityMetrics.consistency} color="text-purple-400" />
              <MetricBar label="Performance" value={nodeData.qualityMetrics.performance} color="text-yellow-400" />
              <MetricBar label="Security" value={nodeData.qualityMetrics.security} color="text-red-400" />
            </div>
          </div>
        )}

        {/* Requirements Section */}
        {nodeData.requirements && nodeData.requirements.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ClipboardDocumentCheckIcon className="w-4 h-4 text-neutral-400" />
                <span className="text-xs font-semibold text-neutral-300 uppercase tracking-wide">
                  Requirements
                </span>
              </div>
            </div>

            {/* Category Filter */}
            <div className="flex flex-wrap gap-2">
              {categories.map(cat => (
                <BubbleButton
                  key={cat}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedCategory(cat);
                  }}
                  variant={selectedCategory === cat ? 'primary' : 'secondary'}
                  className={`
                    px-3 py-1 text-xs font-medium rounded-full transition-colors
                    ${selectedCategory === cat
                      ? 'bg-green-600 text-white'
                      : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                    }
                  `}
                >
                  {cat === 'all' ? 'All' : cat}
                  {cat === 'all' && stats && (
                    <span className="ml-1 opacity-70">({stats.total})</span>
                  )}
                </BubbleButton>
              ))}
            </div>

            {/* Requirements List */}
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {filteredRequirements.map(req => (
                <div
                  key={req.id}
                  className={`
                    bg-neutral-900/50 rounded-lg border transition-all
                    ${req.status === 'pass' ? 'border-green-700/30' :
                      req.status === 'fail' ? 'border-red-700/30' :
                      req.status === 'warning' ? 'border-yellow-700/30' :
                      'border-neutral-700/30'}
                  `}
                >
                  {/* Requirement Header */}
                  <div
                    className="p-3 cursor-pointer hover:bg-neutral-800/50 transition-colors"
                    onClick={() => toggleRequirement(req.id)}
                  >
                    <div className="flex items-start gap-2">
                      <RequirementStatusIcon status={req.status} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h4 className="text-sm font-medium text-neutral-200">
                            {req.name}
                          </h4>
                          <span className="px-1.5 py-0.5 text-[9px] rounded bg-neutral-800 text-neutral-500">
                            {req.category || 'uncategorized'}
                          </span>
                        </div>
                      </div>
                      {expandedRequirements.has(req.id) ? (
                        <ChevronDownIcon className="w-4 h-4 text-neutral-500 flex-shrink-0" />
                      ) : (
                        <ChevronRightIcon className="w-4 h-4 text-neutral-500 flex-shrink-0" />
                      )}
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {expandedRequirements.has(req.id) && (
                    <div className="px-3 pb-3 animate-in slide-in-from-top-1 duration-200">
                      <p className="text-xs text-neutral-400 ml-6">{req.description}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Execution Button */}
        {nodeData.onExecute && nodeData.status !== 'running' && (
          <BubbleButton
            className={`
              w-full px-4 py-2.5 text-sm font-semibold transition-all shadow-md hover:shadow-lg
              ${nodeData.verificationStatus === 'fail' || nodeData.verificationStatus === 'warning'
                ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                : 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white'
              }
            `}
            onClick={(e) => {
              e.stopPropagation();
              nodeData.onExecute?.();
            }}
          >
            <ShieldCheckIcon className="w-4 h-4 inline mr-2" />
            Run Verification
          </BubbleButton>
        )}

        {/* Progress for Running State */}
        {nodeData.status === 'running' && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-400">Verifying solution...</span>
              <span className="text-green-300 font-semibold">
                {typeof nodeData.progress === 'number' ? `${nodeData.progress.toFixed(0)}%` : 'Running checks...'}
              </span>
            </div>
            <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 transition-all duration-300 ease-out animate-pulse"
                style={{ width: `${typeof nodeData.progress === 'number' ? nodeData.progress : 50}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

VerificationNodeComponentBase.displayName = 'VerificationNodeComponent';

export const VerificationNodeComponent = withComponentBoundary(
  VerificationNodeComponentBase,
  'VerificationNodeComponent'
);

export default VerificationNodeComponent;
