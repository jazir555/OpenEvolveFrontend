import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useEffect, useMemo, useState } from 'react';
import type { EvolutionNodeData } from '@/stores/evolutionGraphStore';
import { useEvolutionGraphStore } from '@/stores/evolutionGraphStore';
import { useEvolutionSettingsStore } from '@/stores/evolutionSettingsStore';

const STATUS_STYLES: Record<
  NonNullable<EvolutionNodeData['status']>,
  { border: string; text: string; badge: string; glow: string }
> = {
  survived: {
    border: 'border-emerald-500/70',
    text: 'text-emerald-300',
    badge: 'bg-emerald-500/20 text-emerald-200 border-emerald-500/40',
    glow: 'shadow-[0_0_20px_rgba(16,185,129,0.35)]',
  },
  killed: {
    border: 'border-red-500/70',
    text: 'text-red-300',
    badge: 'bg-red-500/20 text-red-200 border-red-500/40',
    glow: 'shadow-[0_0_20px_rgba(239,68,68,0.25)]',
  },
  evaluating: {
    border: 'border-amber-400/70',
    text: 'text-amber-200',
    badge: 'bg-amber-400/20 text-amber-200 border-amber-400/40',
    glow: 'shadow-[0_0_20px_rgba(251,191,36,0.25)]',
  },
};

export function EvolutionNode({ data, selected }: NodeProps) {
  const nodeData = data as EvolutionNodeData;
  const statusStyle = STATUS_STYLES[nodeData.status ?? 'evaluating'];
  const selectNode = useEvolutionGraphStore((s) => s.selectNode);
  const openModal = useEvolutionGraphStore((s) => s.openModal);
  const cachePreviewsEnabled = useEvolutionSettingsStore((s) =>
    Boolean(s.evolutionInputs.cachePreviews ?? true)
  );
  const [isMitosis, setIsMitosis] = useState(() => {
    if (!nodeData.spawnedAt) return false;
    return Date.now() - nodeData.spawnedAt < 1200;
  });

  useEffect(() => {
    if (!nodeData.spawnedAt) {
      setIsMitosis(false);
      return;
    }
    const elapsed = Date.now() - nodeData.spawnedAt;
    if (elapsed >= 1200) {
      setIsMitosis(false);
      return;
    }
    const timeout = window.setTimeout(() => setIsMitosis(false), 1200 - elapsed);
    return () => window.clearTimeout(timeout);
  }, [nodeData.spawnedAt]);

  const fitnessLabel = useMemo(() => {
    if (nodeData.fitness == null || Number.isNaN(nodeData.fitness)) return '--';
    return nodeData.fitness.toFixed(2);
  }, [nodeData.fitness]);

  const label = nodeData.label || `Variant ${nodeData.generation}`;
  const hasThumbnail = cachePreviewsEnabled && Boolean(nodeData.thumbnailUrl);
  const fitnessValue =
    nodeData.fitness == null || Number.isNaN(nodeData.fitness) ? 0.5 : nodeData.fitness;
  const clampedFitness = Math.max(0, Math.min(1, fitnessValue));
  const scale = 0.9 + clampedFitness * 0.25;
  const isKilled = nodeData.status === 'killed';

  return (
    <div
      className="transition-transform duration-300"
      style={{ transform: `scale(${scale})`, transformOrigin: 'center' }}
    >
      <div
        className={`rounded-2xl border bg-neutral-950/80 p-3 w-[220px] ${statusStyle.border} ${statusStyle.glow} ${
          selected ? 'ring-2 ring-blue-400/60' : ''
        } ${isMitosis ? 'evolution-mitosis' : ''} ${
          isKilled ? 'opacity-40' : 'opacity-100'
        } transition-all duration-300 hover:ring-2 hover:ring-blue-400/40 focus-visible:ring-2 focus-visible:ring-blue-400/70`}
        title={`${label} • Fitness ${fitnessLabel}`}
        data-evolution-tooltip
        data-evolution-node={nodeData.id}
        tabIndex={0}
        role="button"
        aria-label={`Evolution node ${label}`}
        onFocus={() => selectNode(nodeData.id)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            openModal(nodeData.id);
          }
        }}
      >
        <Handle type="target" position={Position.Left} className="opacity-0" />
        <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-neutral-400">
          <span>Gen {nodeData.generation}</span>
          <span className={`px-2 py-0.5 rounded-full border ${statusStyle.badge}`}>
            {nodeData.status}
          </span>
        </div>
        <div className="mt-2 text-sm font-semibold text-neutral-100 truncate">
          {label}
        </div>
        <div className="mt-2 h-[110px] rounded-lg overflow-hidden border border-neutral-800 bg-neutral-900">
          {hasThumbnail ? (
            <img
              src={nodeData.thumbnailUrl ?? undefined}
              alt={label}
              className="h-full w-full object-cover"
            />
          ) : cachePreviewsEnabled ? (
            <div className="h-full w-full flex items-center justify-center text-[11px] text-neutral-500">
              Preview pending
            </div>
          ) : (
            <div className="h-full w-full flex items-center justify-center text-[11px] text-neutral-500">
              Cached preview disabled
            </div>
          )}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-neutral-400">
          <span className={statusStyle.text}>Fitness</span>
          <span className="text-neutral-200 font-semibold">{fitnessLabel}</span>
        </div>
        <Handle type="source" position={Position.Right} className="opacity-0" />
      </div>
    </div>
  );
}
