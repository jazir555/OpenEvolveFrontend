// @ts-nocheck
import React, { memo, useState, useMemo, useEffect } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import {
  LightBulbIcon,
  ChevronDownIcon,
  ChartBarIcon,
  SparklesIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  BeakerIcon,
  CogIcon
} from '@heroicons/react/24/outline';
import type { SolutionNodeData, AlternativeSolution, SolutionMetrics } from '../../types';
import { BubbleButton, BubbleSelect } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

/**
 * SolutionNodeComponent - Specialized node for solution generation
 *
 * Features:
 * - Strategy selector dropdown
 * - Quality score gauge/meter
 * - Confidence indicator with visual meter
 * - Iteration counter
 * - Alternative solutions viewer
 * - Real-time metrics dashboard
 * - Interactive strategy switching
 */
const SolutionNodeComponentBase = memo((props: NodeProps) => {
  const { data, selected } = props;

  // Type assertion to access extended properties
  const nodeData = data as SolutionNodeData;

  const [isExpanded, setIsExpanded] = useState(false);
  const [showAlternatives, setShowAlternatives] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState(
    nodeData.currentStrategy || nodeData.availableStrategies?.[0] || ''
  );

  useEffect(() => {
    const nextStrategy =
      nodeData.currentStrategy || nodeData.availableStrategies?.[0] || '';
    setSelectedStrategy(nextStrategy);
  }, [nodeData.currentStrategy, nodeData.availableStrategies]);

  // Color helpers for quality/confidence
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'from-green-500 to-emerald-500';
    if (score >= 0.6) return 'from-yellow-500 to-orange-500';
    return 'from-red-500 to-rose-500';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const formatPercent = (value: number) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? (numeric * 100).toFixed(0) : '0';
  };

  const formatSeconds = (value: number) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? (numeric / 1000).toFixed(1) : '0.0';
  };

  // Quality score gauge component
  const QualityGauge = ({ score, label }: { score: number; label: string }) => {
    const safeScore = Number.isFinite(score) ? score : 0;
    const percentage = Math.min(Math.max(safeScore * 100, 0), 100);
    const radius = 32;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percentage / 100) * circumference;

    return (
      <div className="flex items-center gap-3">
        <div className="relative w-20 h-20">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="40"
              cy="40"
              r={radius}
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-neutral-800"
            />
            <circle
              cx="40"
              cy="40"
              r={radius}
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              className={`text-${percentage >= 80 ? 'green' : percentage >= 60 ? 'yellow' : 'red'}-500 transition-all duration-500 ease-out`}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-lg font-bold ${percentage >= 80 ? 'text-green-400' : percentage >= 60 ? 'text-yellow-400' : 'text-red-400'}`}>
              {percentage.toFixed(0)}%
            </span>
          </div>
        </div>
        <div>
          <div className="text-xs text-neutral-400">{label}</div>
          <div className="text-sm font-semibold text-neutral-200">Score</div>
        </div>
      </div>
    );
  };

  // Confidence meter
  const ConfidenceMeter = ({ confidence }: { confidence: number }) => {
    const safeConfidence = Number.isFinite(confidence) ? confidence : 0;
    const percentage = safeConfidence * 100;
    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-neutral-400">Confidence</span>
          <span className={`font-semibold ${percentage >= 80 ? 'text-green-400' : percentage >= 60 ? 'text-yellow-400' : 'text-red-400'}`}>
            {percentage.toFixed(0)}%
          </span>
        </div>
        <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
          <div
            className={`h-2 transition-all duration-300 ease-out bg-gradient-to-r ${getScoreColor(safeConfidence)}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  return (
    <div
      className={`
        rounded-lg border-2 transition-all duration-300 cursor-pointer
        bg-purple-950/50 ${selected ? 'border-purple-500 shadow-lg shadow-purple-500/20' : 'border-purple-700 shadow-md'}
        hover:shadow-xl hover:shadow-purple-500/10
        min-w-[380px] max-w-[460px]
      `}
    >
      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="w-3 h-3 bg-purple-500 border-2 border-purple-300"
        style={{ left: -6 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="w-3 h-3 bg-purple-500 border-2 border-purple-300"
        style={{ right: -6 }}
      />

      {/* Header */}
      <div className="p-4 border-b border-purple-800/50">
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center shadow-lg">
            <LightBulbIcon className="w-5 h-5 text-white" />
          </div>

          {/* Title */}
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-purple-100 truncate flex items-center gap-2">
              {nodeData.displayName as any}
              {nodeData.status === 'running' && (
                <SparklesIcon className="w-4 h-4 animate-pulse text-purple-400" />
              )}
            </h3>
            {nodeData.description && (
              <p className="text-xs text-purple-300/70 mt-1 truncate">{nodeData.description as any}</p>
            )}
          </div>

          {/* Iteration Badge */}
          {nodeData.iterations !== undefined && nodeData.iterations > 0 && (
            <div className="flex-shrink-0 px-2 py-1 rounded-lg bg-purple-900/50 border border-purple-700">
              <div className="flex items-center gap-1">
                <ArrowPathIcon className="w-3 h-3 text-purple-400" />
                <span className="text-xs font-bold text-purple-200">{nodeData.iterations}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="p-4 space-y-4">
        {/* Strategy Selector */}
        {nodeData.availableStrategies && nodeData.availableStrategies.length > 0 && (
          <div className="space-y-2">
            <label className="text-xs font-semibold text-purple-300 uppercase tracking-wide">
              Strategy
            </label>
            <div className="relative">
              <BubbleSelect
                value={selectedStrategy}
                onChange={(e) => {
                  e.stopPropagation();
                  setSelectedStrategy(e.target.value);
                  nodeData.onParameterChange?.('strategy', e.target.value);
                }}
                className="w-full pr-10 text-sm bg-neutral-900/50 border border-purple-700 text-purple-100 appearance-none cursor-pointer focus:border-purple-500 focus:outline-none focus:ring-1 focus:ring-purple-500"
              >
                {nodeData.availableStrategies.map(strategy => (
                  <option key={strategy} value={strategy}>
                    {strategy}
                  </option>
                ))}
              </BubbleSelect>
              <CogIcon className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-purple-400 pointer-events-none" />
            </div>
          </div>
        )}

        {/* Quality and Confidence Dashboard */}
        <div className="grid grid-cols-2 gap-4">
          {/* Quality Gauge */}
          {nodeData.qualityScore !== undefined && (
            <QualityGauge score={nodeData.qualityScore} label="Quality" />
          )}

          {/* Confidence Meter */}
          {nodeData.confidence !== undefined && (
            <div className="space-y-3">
              <ConfidenceMeter confidence={nodeData.confidence} />
              {nodeData.iterations !== undefined && (
                <div className="flex items-center gap-2 text-xs bg-neutral-900/30 rounded-lg p-2">
                  <ClockIcon className="w-4 h-4 text-purple-400" />
                  <span className="text-neutral-400">Iteration:</span>
                  <span className="font-semibold text-purple-200">{nodeData.iterations}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Additional Metrics */}
        {nodeData.metrics && (
          <div className="bg-neutral-900/50 rounded-lg p-3 border border-purple-800/30">
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-neutral-400">Convergence:</span>
                <span className="ml-2 font-semibold text-green-400">
                  {formatPercent(nodeData.metrics.convergence)}%
                </span>
              </div>
              <div>
                <span className="text-neutral-400">Diversity:</span>
                <span className="ml-2 font-semibold text-blue-400">
                  {formatPercent(nodeData.metrics.diversity)}%
                </span>
              </div>
              <div>
                <span className="text-neutral-400">Efficiency:</span>
                <span className="ml-2 font-semibold text-purple-400">
                  {formatPercent(nodeData.metrics.efficiency)}%
                </span>
              </div>
              <div>
                <span className="text-neutral-400">Exec Time:</span>
                <span className="ml-2 font-semibold text-yellow-400">
                  {formatSeconds(nodeData.metrics.executionTime)}s
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Alternative Solutions Toggle */}
        {nodeData.alternativeSolutions && nodeData.alternativeSolutions.length > 0 && (
          <div className="space-y-2">
            <BubbleButton
              variant="ghost"
              className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium rounded-lg bg-purple-900/30 text-purple-200 hover:bg-purple-900/50 transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                setShowAlternatives(!showAlternatives);
              }}
            >
              <span className="flex items-center gap-2">
                <BeakerIcon className="w-4 h-4" />
                Alternative Solutions
              </span>
              <span className={`
                px-2 py-0.5 rounded-full text-xs
                ${showAlternatives ? 'bg-purple-600' : 'bg-purple-800'}
              `}>
                {nodeData.alternativeSolutions.length}
              </span>
            </BubbleButton>

            {showAlternatives && (
              <div className="space-y-2 animate-in slide-in-from-top-2 duration-200">
                {nodeData.alternativeSolutions.map((alt, idx) => {
                  const score = alt.score ?? 0;
                  const confidence = alt.confidence ?? 0;
                  return (
                  <div
                    key={alt.id || idx}
                    className={`
                      bg-neutral-900/50 rounded-lg p-3 border transition-all
                      ${score > (nodeData.qualityScore || 0) ? 'border-green-700/50 hover:bg-green-950/20' : 'border-purple-800/30 hover:bg-purple-950/20'}
                    `}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-neutral-300">
                            {alt.name}
                          </span>
                          {score > (nodeData.qualityScore || 0) && (
                            <CheckCircleIcon className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                          )}
                        </div>
                        <div className="text-[10px] text-neutral-500 mt-0.5">{alt.strategy}</div>
                      </div>
                      <div className="flex-shrink-0 text-right">
                        <div className={`text-sm font-bold ${score >= 0.8 ? 'text-green-400' : score >= 0.6 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {(score * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-neutral-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-1.5 bg-gradient-to-r ${getScoreColor(score)}`}
                          style={{ width: `${score * 100}%` }}
                        />
                      </div>
                      <span className="text-[10px] text-neutral-400">
                        {(confidence * 100).toFixed(0)}% conf
                      </span>
                    </div>
                  </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Progress Indicator for Running State */}
        {nodeData.status === 'running' && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-400">Generating solution...</span>
              <span className="text-purple-300 font-semibold">
                {typeof nodeData.progress === 'number' ? `${nodeData.progress.toFixed(0)}%` : 'Processing'}
              </span>
            </div>
            <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 transition-all duration-300 ease-out animate-pulse"
                style={{ width: `${typeof nodeData.progress === 'number' ? nodeData.progress : 50}%` }}
              />
            </div>
          </div>
        )}

        {/* Execution/Retry Button */}
        {nodeData.onExecute && nodeData.status !== 'running' && (
          <BubbleButton
            className={`
              w-full px-4 py-2.5 text-sm font-semibold transition-all shadow-md hover:shadow-lg
              ${nodeData.status === 'error'
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white'
              }
            `}
            onClick={(e) => {
              e.stopPropagation();
              nodeData.onExecute?.();
            }}
          >
            {nodeData.status === 'error' ? (
              <>
                <XCircleIcon className="w-4 h-4 inline mr-2" />
                Retry Solution
              </>
            ) : (
              <>
                <SparklesIcon className="w-4 h-4 inline mr-2" />
                Generate Solution
              </>
            )}
          </BubbleButton>
        )}
      </div>
    </div>
  );
});

SolutionNodeComponentBase.displayName = 'SolutionNodeComponent';

export const SolutionNodeComponent = withComponentBoundary(
  SolutionNodeComponentBase,
  'SolutionNodeComponent'
);

export default SolutionNodeComponent;
