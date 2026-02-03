import { useMemo } from 'react';
import {
  useEvolutionGraphStore,
  type EvolutionGraphNode,
  type EvolutionNodeData,
} from '@/stores/evolutionGraphStore';
import { useEvolutionSettingsStore } from '@/stores/evolutionSettingsStore';

const STATUS_STYLES: Record<
  EvolutionNodeData['status'],
  { badge: string; text: string }
> = {
  survived: {
    badge: 'bg-emerald-500/20 text-emerald-200 border-emerald-500/40',
    text: 'text-emerald-200',
  },
  killed: {
    badge: 'bg-red-500/20 text-red-200 border-red-500/40',
    text: 'text-red-200',
  },
  evaluating: {
    badge: 'bg-amber-400/20 text-amber-200 border-amber-400/40',
    text: 'text-amber-200',
  },
};

export function EvolutionInspectorPanel() {
  const nodes = useEvolutionGraphStore((s) => s.nodes);
  const selectedNodeId = useEvolutionGraphStore((s) => s.selectedNodeId);
  const selectNode = useEvolutionGraphStore((s) => s.selectNode);
  const previewMode = useEvolutionGraphStore((s) => s.previewMode);
  const setPreviewMode = useEvolutionGraphStore((s) => s.setPreviewMode);
  const openModal = useEvolutionGraphStore((s) => s.openModal);
  const cachePreviewsEnabled = useEvolutionSettingsStore((s) =>
    Boolean(s.evolutionInputs.cachePreviews ?? true)
  );

  const selectedNode = useMemo(
    () => nodes.find((node) => node.data.id === selectedNodeId)?.data ?? null,
    [nodes, selectedNodeId]
  );

  if (!selectedNode) {
    return (
      <div className="h-full w-full bg-[#0b0f1a] border-l border-neutral-800 flex items-center justify-center text-center px-6">
        <div>
          <h3 className="text-sm uppercase tracking-[0.2em] text-neutral-400">
            Inspector
          </h3>
          <p className="mt-3 text-sm text-neutral-300">
            Select a node to inspect its live render, cached preview, and
            metadata.
          </p>
        </div>
      </div>
    );
  }

  const statusStyle = STATUS_STYLES[selectedNode.status ?? 'evaluating'];
  const fitnessLabel =
    selectedNode.fitness == null || Number.isNaN(selectedNode.fitness)
      ? '--'
      : selectedNode.fitness.toFixed(2);
  const scoreLabel =
    selectedNode.score == null || Number.isNaN(selectedNode.score)
      ? '--'
      : selectedNode.score.toFixed(2);

  const hasHtml = Boolean(selectedNode.htmlUrl || selectedNode.html);
  const hasCached = cachePreviewsEnabled && Boolean(selectedNode.thumbnailUrl);

  return (
    <div className="h-full w-full bg-[#0b0f1a] border-l border-neutral-800 flex flex-col">
      <div className="px-5 py-4 border-b border-neutral-800 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-neutral-400">
            Inspector
          </p>
          <h3 className="text-base font-semibold text-white">
            {selectedNode.label || `Generation ${selectedNode.generation}`}
          </h3>
        </div>
        <button
          type="button"
          onClick={() => selectNode(null)}
          className="text-xs text-neutral-400 hover:text-white transition-colors"
        >
          Clear
        </button>
      </div>

      <div className="px-5 py-4 border-b border-neutral-800 space-y-3">
        <div className="flex items-center justify-between text-xs text-neutral-400">
          <span>Generation {selectedNode.generation}</span>
          <span
            className={`px-2 py-1 rounded-full border text-[10px] uppercase tracking-[0.18em] ${statusStyle.badge}`}
          >
            {selectedNode.status}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs text-neutral-400">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
              Fitness
            </p>
            <p className={`text-sm font-semibold ${statusStyle.text}`}>
              {fitnessLabel}
            </p>
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
              Score
            </p>
            <p className="text-sm font-semibold text-neutral-100">
              {scoreLabel}
            </p>
          </div>
        </div>
      </div>

      <div className="px-5 py-4 border-b border-neutral-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-neutral-400">
            <span>Preview Mode</span>
          </div>
          <div className="flex rounded-full border border-neutral-700 overflow-hidden text-xs">
            {(['live', 'cached'] as const).map((mode) => {
              const isDisabled = mode === 'cached' && !cachePreviewsEnabled;
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setPreviewMode(mode)}
                  disabled={isDisabled}
                  aria-pressed={previewMode === mode}
                  aria-label={`Preview mode ${mode}`}
                  className={`px-3 py-1.5 uppercase tracking-[0.16em] ${
                    previewMode === mode
                      ? 'bg-blue-500/20 text-blue-200'
                      : 'text-neutral-400 hover:text-neutral-200'
                  } ${isDisabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                >
                  {mode}
                </button>
              );
            })}
          </div>
        </div>
        {!cachePreviewsEnabled && (
          <p className="mt-2 text-[11px] text-neutral-500">
            Cached previews are disabled in settings.
          </p>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-auto p-5">
        {previewMode === 'live' ? (
          hasHtml ? (
            <iframe
              title={selectedNode.label || 'Evolution preview'}
              src={selectedNode.htmlUrl || undefined}
              srcDoc={
                selectedNode.htmlUrl ? undefined : selectedNode.html || undefined
              }
              className="w-full h-[420px] rounded-xl border border-neutral-800 bg-white"
              sandbox="allow-scripts allow-same-origin"
            />
          ) : (
            <div className="h-[320px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm">
              Live HTML preview not available yet.
            </div>
          )
        ) : !cachePreviewsEnabled ? (
          <div className="h-[320px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm">
            Cached previews are disabled in settings.
          </div>
        ) : hasCached ? (
          <img
            src={selectedNode.thumbnailUrl ?? undefined}
            alt={selectedNode.label || 'Cached preview'}
            className="w-full rounded-xl border border-neutral-800"
          />
        ) : (
          <div className="h-[320px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm">
            Cached preview not available yet.
          </div>
        )}
      </div>

      <div className="px-5 py-4 border-t border-neutral-800">
        <button
          type="button"
          onClick={() => openModal(selectedNode.id)}
          className="w-full px-3 py-2 rounded-lg bg-blue-500/20 text-blue-200 text-xs uppercase tracking-[0.16em] border border-blue-500/40"
        >
          Open Full Preview
        </button>
      </div>
    </div>
  );
}
