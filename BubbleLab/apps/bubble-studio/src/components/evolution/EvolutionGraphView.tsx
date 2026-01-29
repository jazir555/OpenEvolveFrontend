import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  type ImperativePanelHandle,
} from 'react-resizable-panels';
import { toast } from 'react-toastify';
import { useEvolutionGraphStore } from '@/stores/evolutionGraphStore';
import type { EvolutionGraphNode, EvolutionNodeData } from '@/stores/evolutionGraphStore';
import { useEvolutionSettingsStore } from '@/stores/evolutionSettingsStore';
import { evolutionGraphApi } from '@/services/evolutionGraphApi';
import { API_BASE_URL } from '@/env';
import { EvolutionNode } from '@/components/evolution/EvolutionNode';
import { EvolutionPreviewModal } from '@/components/evolution/EvolutionPreviewModal';
import { EvolutionInspectorPanel } from '@/components/evolution/EvolutionInspectorPanel';

const NODE_TYPES = {
  evolution: EvolutionNode,
};

const STATUS_COLORS: Record<string, string> = {
  survived: '#34d399',
  killed: '#f87171',
  evaluating: '#fbbf24',
};

const X_SPACING = 320;
const Y_SPACING = 220;

export function EvolutionGraphView() {
  const runId = useEvolutionGraphStore((s) => s.runId);
  const evolutionId = useEvolutionGraphStore((s) => s.evolutionId);
  const nodes = useEvolutionGraphStore((s) => s.nodes);
  const selectedNodeId = useEvolutionGraphStore((s) => s.selectedNodeId);
  const setActiveRun = useEvolutionGraphStore((s) => s.setActiveRun);
  const setNodes = useEvolutionGraphStore((s) => s.setNodes);
  const selectNode = useEvolutionGraphStore((s) => s.selectNode);
  const modalNodeId = useEvolutionGraphStore((s) => s.modalNodeId);
  const openModal = useEvolutionGraphStore((s) => s.openModal);
  const closeModal = useEvolutionGraphStore((s) => s.closeModal);
  const previewMode = useEvolutionGraphStore((s) => s.previewMode);
  const setPreviewMode = useEvolutionGraphStore((s) => s.setPreviewMode);
  const cachePreviewsEnabled = useEvolutionSettingsStore((s) =>
    Boolean(s.evolutionInputs.cachePreviews ?? true)
  );
  const inspectorRef = useRef<ImperativePanelHandle>(null);
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoverPosition, setHoverPosition] = useState<{ x: number; y: number } | null>(
    null
  );
  const [visibleGeneration, setVisibleGeneration] = useState<number | null>(null);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);

  useEffect(() => {
    if (runId) return;
    let cancelled = false;

    evolutionGraphApi
      .listRuns()
      .then((runs) => {
        if (cancelled || runs.length === 0) return;
        const latest = runs[0];
        setActiveRun(latest.id, latest.evolutionId);
      })
      .catch(() => {
        // No-op: evolution graph can stay empty until a run starts.
      });

    return () => {
      cancelled = true;
    };
  }, [runId, setActiveRun]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;

    const resolveAssetUrl = (assetId?: number | null) => {
      if (!assetId) return null;
      return `${API_BASE_URL}/evolution-graph/assets/${assetId}`;
    };

    evolutionGraphApi
      .listNodes(runId)
      .then((records) => {
        if (cancelled) return;
        setNodes(
          records.map((record) => ({
            id: record.nodeId,
            parentId: record.parentNodeId ?? null,
            generation: record.generation,
            status: record.status as 'evaluating' | 'survived' | 'killed',
            fitness: record.fitness ?? null,
            score: record.score ?? null,
            label: record.label ?? undefined,
            htmlUrl: resolveAssetUrl(record.htmlAssetId),
            thumbnailUrl: resolveAssetUrl(record.thumbnailAssetId),
            metadata: record.metadata ?? undefined,
            createdAt: record.createdAt,
          }))
        );
      })
      .catch(() => {
        // Ignore load errors; realtime updates will repopulate.
      });

    return () => {
      cancelled = true;
    };
  }, [runId, setNodes]);

  useEffect(() => {
    if (!cachePreviewsEnabled && previewMode === 'cached') {
      setPreviewMode('live');
    }
  }, [cachePreviewsEnabled, previewMode, setPreviewMode]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);
      if (isTyping) return;

      if (event.key === 'Escape' && modalNodeId) {
        event.preventDefault();
        closeModal();
        return;
      }

      if (
        (event.key === 'Enter' || event.key === ' ') &&
        selectedNodeId &&
        !modalNodeId
      ) {
        event.preventDefault();
        openModal(selectedNodeId);
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedNodeId, modalNodeId, openModal, closeModal]);

  const maxGeneration = useMemo(() => {
    if (!nodes.length) return 0;
    return Math.max(...nodes.map((node) => node.data.generation));
  }, [nodes]);

  useEffect(() => {
    if (!nodes.length) return;
    setVisibleGeneration((current) =>
      current == null || current > maxGeneration ? maxGeneration : current
    );
  }, [nodes.length, maxGeneration]);

  useEffect(() => {
    if (!isTimelinePlaying || visibleGeneration == null) return;
    const timer = window.setInterval(() => {
      setVisibleGeneration((current) => {
        if (current == null) return current;
        if (current >= maxGeneration) {
          setIsTimelinePlaying(false);
          return current;
        }
        return current + 1;
      });
    }, playbackSpeed);
    return () => window.clearInterval(timer);
  }, [isTimelinePlaying, playbackSpeed, maxGeneration, visibleGeneration]);

  const filteredNodes = useMemo(() => {
    const generation = visibleGeneration ?? maxGeneration;
    return nodes.filter((node) => node.data.generation <= generation);
  }, [nodes, visibleGeneration, maxGeneration]);

  const graphNodes = useMemo(() => {
    if (!filteredNodes.length) return [] as Node[];

    const groups = new Map<number, EvolutionGraphNode[]>();
    for (const node of filteredNodes) {
      const group = groups.get(node.data.generation) ?? [];
      group.push(node);
      groups.set(node.data.generation, group);
    }

    const generations = Array.from(groups.keys()).sort((a, b) => a - b);
    const generationIndex = new Map<number, number>();
    generations.forEach((generation, index) => {
      generationIndex.set(generation, index);
    });

    return filteredNodes.map((node) => {
      const index = generationIndex.get(node.data.generation) ?? 0;
      const siblings = groups.get(node.data.generation) ?? [];
      const sortedSiblings = [...siblings].sort((a, b) => {
        const timeA = a.data.createdAt ? Date.parse(a.data.createdAt) : 0;
        const timeB = b.data.createdAt ? Date.parse(b.data.createdAt) : 0;
        if (timeA !== timeB) return timeA - timeB;
        return a.data.id.localeCompare(b.data.id);
      });
      const siblingIndex = sortedSiblings.findIndex((item) => item.data.id === node.data.id);
      const yOffset = -((sortedSiblings.length - 1) * Y_SPACING) / 2;

      return {
        id: node.id,
        type: 'evolution',
        data: node.data,
        position: {
          x: index * X_SPACING,
          y: yOffset + siblingIndex * Y_SPACING,
        },
      };
    });
  }, [filteredNodes]);

  const graphEdges = useMemo(() => {
    const nodeIds = new Set(filteredNodes.map((node) => node.data.id));
    return filteredNodes
      .filter((node) => node.data.parentId && nodeIds.has(String(node.data.parentId)))
      .map((node) => {
        const color = STATUS_COLORS[node.data.status] ?? '#94a3b8';
        return {
          id: `${node.data.parentId}-${node.data.id}`,
          source: String(node.data.parentId),
          target: node.data.id,
          type: 'smoothstep',
          animated: node.data.status === 'evaluating',
          style: { stroke: color, strokeWidth: 2 },
        } as Edge;
      });
  }, [filteredNodes]);

  const modalNode = nodes.find((node) => node.data.id === modalNodeId) ?? null;
  const hoveredNode = nodes.find((node) => node.data.id === hoveredNodeId) ?? null;

  const downloadHtml = async (nodeData: EvolutionNodeData) => {
    let html = nodeData.html ?? null;
    if (!html && nodeData.htmlUrl) {
      try {
        const response = await fetch(nodeData.htmlUrl);
        html = await response.text();
      } catch {
        toast.error('Failed to download HTML for this node.');
        return;
      }
    }
    if (!html) {
      toast.info('No HTML available for export.');
      return;
    }
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${nodeData.id}.html`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('HTML download started.');
  };

  const updateHoverPosition = (event: React.MouseEvent) => {
    if (!graphContainerRef.current) return;
    const rect = graphContainerRef.current.getBoundingClientRect();
    setHoverPosition({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  };

  return (
    <div className="h-full w-full bg-gradient-to-br from-[#090d16] via-[#0b1220] to-[#0a0a0f]">
      <PanelGroup
        direction="horizontal"
        autoSaveId="evolution-graph-layout"
        className="h-full w-full"
      >
        <Panel defaultSize={70} minSize={45} className="relative">
          <div className="h-full w-full relative" ref={graphContainerRef}>
            <ReactFlow
              nodes={graphNodes}
              edges={graphEdges}
              nodeTypes={NODE_TYPES}
              fitView
              nodesDraggable={false}
              nodesConnectable={false}
              nodesFocusable
              onNodeClick={(_: React.MouseEvent, node: Node) => {
                if (selectedNodeId === node.id && !modalNodeId) {
                  openModal(node.id);
                  return;
                }
                selectNode(node.id);
              }}
              onNodeDoubleClick={(_: React.MouseEvent, node: Node) => {
                openModal(node.id);
                const target = node.data as unknown as EvolutionNodeData;
                void downloadHtml(target);
              }}
              onNodeMouseEnter={(event: React.MouseEvent, node: Node) => {
                setHoveredNodeId(node.id);
                updateHoverPosition(event);
              }}
              onNodeMouseMove={(event: React.MouseEvent) => {
                if (hoveredNodeId) {
                  updateHoverPosition(event);
                }
              }}
              onNodeMouseLeave={() => {
                setHoveredNodeId(null);
                setHoverPosition(null);
              }}
              onPaneClick={() => selectNode(null)}
            >
              <Background gap={32} size={1} color="#1f2937" />
              <Controls showInteractive={false} />
              <MiniMap
                nodeColor={(node: Node) => {
                  const data = node.data as unknown as EvolutionNodeData;
                  return STATUS_COLORS[data.status ?? 'evaluating'] ?? '#94a3b8';
                }}
              />
            </ReactFlow>

            {!runId && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 px-6 py-5 text-center">
                  <h3 className="text-lg font-semibold text-white">No evolution run</h3>
                  <p className="text-sm text-neutral-400 mt-2">
                    Start an evolution run to watch the lineage grow.
                  </p>
                </div>
              </div>
            )}

            {runId && nodes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 px-6 py-5 text-center">
                  <h3 className="text-lg font-semibold text-white">Awaiting descendants</h3>
                  <p className="text-sm text-neutral-400 mt-2">
                    Evolution {evolutionId ? `#${evolutionId}` : ''} is running.
                  </p>
                </div>
              </div>
            )}

            {isCollapsed && (
              <button
                type="button"
                onClick={() => inspectorRef.current?.expand(30)}
                className="absolute top-4 right-4 z-10 px-3 py-2 rounded-lg bg-[#101826] border border-blue-500/40 text-blue-200 text-xs uppercase tracking-wide shadow-lg"
              >
                Show Inspector
              </button>
            )}

            {nodes.length > 0 && (
              <div className="absolute bottom-4 left-4 z-10 rounded-xl border border-neutral-800 bg-neutral-950/90 px-4 py-3 text-xs text-neutral-300 shadow-xl">
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setIsTimelinePlaying((value) => !value)}
                    className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors"
                  >
                    {isTimelinePlaying ? 'Pause' : 'Play'}
                  </button>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-neutral-500">Generation</span>
                    <input
                      type="range"
                      min={0}
                      max={maxGeneration}
                      value={visibleGeneration ?? maxGeneration}
                      aria-label="Generation scrubber"
                      onChange={(event) => {
                        setIsTimelinePlaying(false);
                        setVisibleGeneration(Number(event.target.value));
                      }}
                      className="w-36 accent-blue-500"
                    />
                    <span className="text-[11px] text-neutral-400">
                      {visibleGeneration ?? maxGeneration}/{maxGeneration}
                    </span>
                  </div>
                  <select
                    value={String(playbackSpeed)}
                    aria-label="Timeline playback speed"
                    onChange={(event) =>
                      setPlaybackSpeed(Number(event.target.value))
                    }
                    className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs text-neutral-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="2000">0.5x</option>
                    <option value="1000">1x</option>
                    <option value="500">2x</option>
                  </select>
                </div>
              </div>
            )}

            {hoveredNode && hoverPosition && (
              <div
                className="absolute z-20 pointer-events-none"
                style={{
                  left: Math.min(hoverPosition.x + 16, 720),
                  top: Math.min(hoverPosition.y + 16, 520),
                }}
              >
                <div className="w-64 rounded-xl border border-neutral-800 bg-neutral-950/95 shadow-xl overflow-hidden">
                  <div className="px-3 py-2 border-b border-neutral-800">
                    <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                      Hover Preview
                    </p>
                    <p className="text-xs font-semibold text-white truncate">
                      {hoveredNode.data.label || `Generation ${hoveredNode.data.generation}`}
                    </p>
                  </div>
                  <div className="h-36 bg-neutral-900 flex items-center justify-center">
                    {hoveredNode.data.thumbnailUrl ? (
                      <img
                        src={hoveredNode.data.thumbnailUrl}
                        alt={hoveredNode.data.label || 'Preview'}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <span className="text-[11px] text-neutral-500 px-4 text-center">
                        {previewMode === 'live'
                          ? 'Live preview available in the inspector.'
                          : 'Preview not cached yet.'}
                      </span>
                    )}
                  </div>
                  <div className="px-3 py-2 text-[11px] text-neutral-400 flex items-center justify-between">
                    <span>Status: {hoveredNode.data.status}</span>
                    <span>
                      Fitness {hoveredNode.data.fitness?.toFixed(2) ?? '--'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Panel>
        <PanelResizeHandle className="w-2 bg-[#30363d] hover:bg-white/70 transition-colors" />
        <Panel
          ref={inspectorRef}
          defaultSize={30}
          minSize={20}
          maxSize={45}
          collapsible
          collapsedSize={0}
          onCollapse={() => setIsCollapsed(true)}
          onExpand={() => setIsCollapsed(false)}
        >
          <EvolutionInspectorPanel />
        </Panel>
      </PanelGroup>

      {modalNode && (
        <EvolutionPreviewModal
          key={modalNode.data.id}
          node={modalNode.data}
          onClose={closeModal}
        />
      )}
    </div>
  );
}
