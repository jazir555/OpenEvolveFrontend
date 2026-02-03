import { create } from 'zustand';
import type { Node } from '@xyflow/react';

export type EvolutionNodeStatus = 'evaluating' | 'survived' | 'killed';

/**
 * Evolution node data interface
 * This is the data structure stored in the EvolutionGraphNode.data property
 * Includes index signature for compatibility with @xyflow/react's Node type
 */
export interface EvolutionNodeData extends Record<string, unknown> {
  id: string;
  parentId?: string | null;
  generation: number;
  status: EvolutionNodeStatus;
  fitness?: number | null;
  score?: number | null;
  label?: string | null;
  html?: string | null;
  htmlUrl?: string | null;
  thumbnailUrl?: string | null;
  metadata?: Record<string, unknown> | null;
  createdAt?: string;
  spawnedAt?: number;
}

/**
 * Full graph node type compatible with @xyflow/react
 * This extends the Node type and uses EvolutionNodeData for the data property
 */
export interface EvolutionGraphNode extends Node {
  data: EvolutionNodeData;
}

export type EvolutionPreviewMode = 'live' | 'cached';

interface EvolutionGraphState {
  runId: number | null;
  evolutionId: string | null;
  nodes: EvolutionGraphNode[];
  selectedNodeId: string | null;
  modalNodeId: string | null;
  previewMode: EvolutionPreviewMode;
  setActiveRun: (runId: number | null, evolutionId: string | null) => void;
  setNodes: (nodes: EvolutionNodeData[]) => void;
  addNode: (node: EvolutionNodeData) => void;
  updateNode: (nodeId: string, patch: Partial<EvolutionNodeData>) => void;
  clearGraph: () => void;
  selectNode: (nodeId: string | null) => void;
  openModal: (nodeId: string) => void;
  closeModal: () => void;
  setPreviewMode: (mode: EvolutionPreviewMode) => void;
}

const normalizeNode = (node: EvolutionNodeData): EvolutionNodeData => ({
  ...node,
  status: node.status ?? 'evaluating',
  spawnedAt: node.spawnedAt ?? Date.now(),
});

export const useEvolutionGraphStore = create<EvolutionGraphState>((set) => ({
  runId: null,
  evolutionId: null,
  nodes: [],
  selectedNodeId: null,
  modalNodeId: null,
  previewMode: 'live',
  setActiveRun: (runId, evolutionId) =>
    set({ runId, evolutionId, selectedNodeId: null, modalNodeId: null }),
  setNodes: (nodesData) =>
    set({
      nodes: nodesData.map((data) => ({
        id: data.id,
        type: 'evolution',
        data: normalizeNode(data),
        position: { x: 0, y: 0 },
      })),
      selectedNodeId: null,
      modalNodeId: null,
    }),
  addNode: (nodeData) =>
    set((state) => {
      const normalized = normalizeNode(nodeData);
      const existingIndex = state.nodes.findIndex((item) => item.data.id === normalized.id);
      if (existingIndex >= 0) {
        const next = [...state.nodes];
        next[existingIndex] = {
          ...next[existingIndex],
          data: { ...next[existingIndex].data, ...normalized },
        };
        return { nodes: next };
      }
      return {
        nodes: [
          ...state.nodes,
          {
            id: normalized.id,
            type: 'evolution',
            data: normalized,
            position: { x: 0, y: 0 },
          },
        ],
      };
    }),
  updateNode: (nodeId, patch) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.data.id === nodeId ? { ...node, data: { ...node.data, ...patch } } : node
      ),
    })),
  clearGraph: () => set({ nodes: [], selectedNodeId: null, modalNodeId: null }),
  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),
  openModal: (nodeId) => set({ modalNodeId: nodeId }),
  closeModal: () => set({ modalNodeId: null }),
  setPreviewMode: (mode) => set({ previewMode: mode }),
}));
