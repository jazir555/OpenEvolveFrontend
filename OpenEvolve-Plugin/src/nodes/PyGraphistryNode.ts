// @ts-nocheck
/**
 * PyGraphistry Node - Integration Library Version
 *
 * This node uses the OpenEvolve Integration Library to communicate with
 * the PyGraphistry visualization service, which generates interactive graph visualizations.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { apiClient } from '@/services/api';
import { useAuthStore } from '@/stores/authStore';

/**
 * PyGraphistry visualization layout types
 */
export type PyGraphistryLayout = 'force_directed' | 'circular' | 'hierarchical';

/**
 * PyGraphistry clustering methods
 */
export type PyGraphistryClusteringMethod = 'dbscan' | 'kmeans';

/**
 * Graph node interface
 */
export interface GraphNode {
  id: string;
  label?: string;
  type?: string;
  [key: string]: string | number | boolean | undefined;
}

/**
 * Graph edge interface
 */
export interface GraphEdge {
  source: string;
  target: string;
  type?: string;
  weight?: number;
  [key: string]: string | number | boolean | undefined;
}

/**
 * PyGraphistry node configuration
 */
export interface PyGraphistryNodeConfig {
  layout?: PyGraphistryLayout;
  clustering?: boolean;
  clusteringMethod?: PyGraphistryClusteringMethod;
  enableGPUAcceleration?: boolean;
  apiKey?: string;
  serverUrl?: string;
  enableBackendExecution?: boolean;
  backendUrl?: string;
}

/**
 * PyGraphistry result interface
 */
export interface PyGraphistryResult {
  success: boolean;
  visualizationUrl?: string;
  message: string;
  metadata: {
    executionTime: number;
    backendUsed: boolean;
    nodesProcessed: number;
    edgesProcessed: number;
    layoutUsed: PyGraphistryLayout;
    clusteringApplied: boolean;
  };
  error?: string;
}

/**
 * PyGraphistry Node (Integration Library Version)
 *
 * This node uses the OpenEvolve Integration Library to delegate visualization
 * to the PyGraphistry service. The PyGraphistry service generates interactive
 * graph visualizations from node and edge data.
 *
 * Benefits of this approach:
 * - Reuses existing PyGraphistry visualization capabilities
 * - No need to duplicate logic in TypeScript
 * - Consistent behavior across all clients
 * - Easy to update backend without changing frontend
 */
export class PyGraphistryNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'PyGraphistry Visualization';
  static readonly DESCRIPTION = 'Generate interactive graph visualizations using PyGraphistry via integration library';
  static readonly ICON = 'graph';
  static readonly CATEGORY = 'visualization';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: PyGraphistryNodeConfig = {}) {
    super(id, {
      layout: 'force_directed',
      clustering: false,
      clusteringMethod: 'dbscan',
      enableGPUAcceleration: true,
      serverUrl: 'http://localhost:8000',
      enableBackendExecution: true,
      backendUrl: 'http://localhost:8000',
      ...config
    });
  }

  /**
   * Execute PyGraphistry visualization using the integration library
   *
   * @param inputs - Must contain 'nodes' and 'edges' arrays
   * @param context - Execution context
   * @returns Promise resolving to visualization result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Validate inputs first
      const validationErrors = this.validateInputs(inputs);
      if (validationErrors.length > 0) {
        return this.createErrorResult(
          `Validation failed: ${validationErrors.map(e => e.message).join('; ')}`
        );
      }

      // Extract inputs with additional validation
      let nodes = inputs.nodes as GraphNode[] | undefined;
      let edges = inputs.edges as GraphEdge[] | undefined;
      const layout = (inputs.layout as PyGraphistryLayout) || this.config.layout;
      const clustering = inputs.clustering as boolean | undefined;
      const clusteringMethod = (inputs.clusteringMethod as PyGraphistryClusteringMethod) || this.config.clusteringMethod;

      // Ensure nodes and edges are arrays and not null/undefined
      if (!nodes || !Array.isArray(nodes)) {
        nodes = [];
      }
      if (!edges || !Array.isArray(edges)) {
        edges = [];
      }

      // Additional safeguard: ensure nodes and edges have reasonable limits
      if (nodes.length > 10000) {
        console.warn(`Large number of nodes (${nodes.length}) detected, consider reducing for performance`);
      }
      if (edges.length > 50000) {
        console.warn(`Large number of edges (${edges.length}) detected, consider reducing for performance`);
      }

      context.updateProgress(10, 'Validating inputs');

      // Use integration library to call PyGraphistry service
      if (this.config.enableBackendExecution) {
        try {
          return await this.executeWithBackend(
            nodes,
            edges,
            layout,
            clustering ?? this.config.clustering,
            clusteringMethod,
            context
          );
        } catch (backendError) {
          console.warn('PyGraphistry backend execution failed, falling back to local execution:', backendError);
          context.updateProgress(20, 'Backend unavailable, using local execution');
          return await this.executeLocally(
            nodes,
            edges,
            layout,
            clustering ?? this.config.clustering,
            context
          );
        }
      } else {
        return await this.executeLocally(
          nodes,
          edges,
          layout,
          clustering ?? this.config.clustering,
          context
        );
      }

    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'PyGraphistryNode execution error' } });
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during PyGraphistry visualization'
      );
    }
  }

  /**
   * Execute visualization using PyGraphistry service via integration library
   */
  private async executeWithBackend(
    nodes: GraphNode[],
    edges: GraphEdge[],
    layout: PyGraphistryLayout,
    clustering: boolean,
    clusteringMethod: PyGraphistryClusteringMethod,
    context: ExecutionContext
  ): Promise<NodeResult> {
    const startTime = Date.now(); // Capture start time

    try {
      context.updateProgress(20, 'Connecting to PyGraphistry service');

      // Validate inputs before sending to backend
      if (!nodes || !Array.isArray(nodes) || nodes.length === 0) {
        throw new Error('Nodes array is required and cannot be empty');
      }

      if (!edges || !Array.isArray(edges)) {
        throw new Error('Edges array is required');
      }

      // Prepare request for backend with additional validation
      const backendInputs: Record<string, any> = {
        nodes: nodes.slice(0, 10000), // Cap nodes to prevent overload
        edges: edges.slice(0, 50000), // Cap edges to prevent overload
        layout,
        clustering,
        clusteringMethod,
        config: {
          apiKey: this.config.apiKey,
          serverUrl: this.config.serverUrl,
          enableGPUAcceleration: this.config.enableGPUAcceleration
        }
      };

      context.updateProgress(30, 'Generating visualization on backend');

      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

      try {
        const result: PyGraphistryResult = await this.postToBackend('/visualize/pygraphistry', backendInputs);

        clearTimeout(timeoutId);

        context.updateProgress(100, 'Visualization complete');

        // Validate and transform backend result to match expected output format
        if (!result) {
          throw new Error('Invalid response from PyGraphistry service');
        }

        const transformedResult = {
          visualizationUrl: result.visualizationUrl || null,
          message: result.message || 'Visualization generated successfully',
          metadata: {
            executionTime: Date.now() - startTime,
            backendUsed: true,
            nodesProcessed: Math.min(nodes.length, 10000),
            edgesProcessed: Math.min(edges.length, 50000),
            layoutUsed: layout,
            clusteringApplied: clustering,
            ...result.metadata
          },
          success: result.success ?? false,
          error: result.error
        };

        return this.createSuccessResult(transformedResult);
      } catch (requestError) {
        clearTimeout(timeoutId);

        // Check if it was a timeout error
        if (requestError.name === 'AbortError' || (requestError instanceof Error && requestError.message.includes('timeout'))) {
          throw new Error('PyGraphistry service request timed out after 60 seconds');
        }

        throw requestError; // Re-throw other errors
      }

    } catch (error) {
      // If backend call fails, fall back to local execution
      console.warn('PyGraphistry service execution failed, falling back to local:', error);
      context.updateProgress(20, 'PyGraphistry service unavailable, using local execution');
      return this.executeLocally(nodes, edges, layout, clustering, clusteringMethod, context);
    }
  }

  /**
   * Execute visualization locally (production version)
   * This processes the graph data using local graph algorithms when backend is unavailable
   */
  private async executeLocally(
    nodes: GraphNode[],
    edges: GraphEdge[],
    layout: PyGraphistryLayout,
    clustering: boolean,
    context: ExecutionContext
  ): Promise<NodeResult> {
    try {
      const startTime = Date.now(); // Capture start time
      context.updateProgress(40, 'Performing local visualization processing');

      // Validate inputs for local execution
      if (!nodes || !Array.isArray(nodes)) {
        nodes = [];
      }
      if (!edges || !Array.isArray(edges)) {
        edges = [];
      }

      // Sanitize inputs
      const sanitizedNodeCount = Math.min(nodes.length, 10000);
      const sanitizedEdgeCount = Math.min(edges.length, 50000);

      // Process the graph data using local algorithms
      const processedGraph = await this.processGraphLocally(
        nodes.slice(0, sanitizedNodeCount),
        edges.slice(0, sanitizedEdgeCount),
        layout,
        clustering
      );

      // Generate a visualization URL based on the processed graph
      const visualizationUrl = await this.generateLocalVisualization(processedGraph, layout);

      context.updateProgress(100, 'Local visualization processing complete');

      return this.createSuccessResult({
        visualizationUrl: visualizationUrl,
        message: 'Local visualization generated successfully',
        metadata: {
          executionTime: Date.now() - startTime, // Calculate actual time taken
          backendUsed: false,
          nodesProcessed: sanitizedNodeCount,
          edgesProcessed: sanitizedEdgeCount,
          layoutUsed: layout,
          clusteringApplied: clustering,
          note: 'Executed locally using production graph algorithms'
        },
        success: true
      });
    } catch (localError) {
      errorLogger.logError(localError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in local execution' } });
      return this.createErrorResult(
        localError instanceof Error ? localError.message : 'Unknown error during local execution'
      );
    }
  }

  /**
   * Process graph data using local algorithms
   */
  private async processGraphLocally(
    nodes: GraphNode[],
    edges: GraphEdge[],
    layout: PyGraphistryLayout,
    clustering: boolean
  ): Promise<{ nodes: any[], edges: any[], positions: Record<string, { x: number; y: number }>, clusters?: any }> {
    // Create a comprehensive graph representation
    const graph = {
      nodes: nodes.map(node => ({ ...node })),
      edges: edges.map(edge => ({ ...edge })),
      adjacency: new Map<string, string[]>(),
      weights: new Map<string, number>(), // Store edge weights
      degrees: new Map<string, number>() // Store node degrees
    };

    // Build adjacency list and calculate degrees
    for (const edge of edges) {
      if (!graph.adjacency.has(edge.source)) {
        graph.adjacency.set(edge.source, []);
        graph.degrees.set(edge.source, 0);
      }
      if (!graph.adjacency.has(edge.target)) {
        graph.adjacency.set(edge.target, []);
        graph.degrees.set(edge.target, 0);
      }

      graph.adjacency.get(edge.source)!.push(edge.target);
      graph.adjacency.get(edge.target)!.push(edge.source);

      // Update degrees
      graph.degrees.set(edge.source, (graph.degrees.get(edge.source) || 0) + 1);
      graph.degrees.set(edge.target, (graph.degrees.get(edge.target) || 0) + 1);

      // Store edge weight
      const edgeKey = `${edge.source}-${edge.target}`;
      graph.weights.set(edgeKey, edge.weight || 1);
    }

    // Initialize all nodes with degree 0 if not already set
    for (const node of nodes) {
      if (!graph.degrees.has(node.id)) {
        graph.degrees.set(node.id, 0);
      }
    }

    // Calculate positions based on layout algorithm
    let positions: Record<string, { x: number; y: number }> = {};

    switch (layout) {
      case 'force_directed':
        positions = await this.calculateForceDirectedPositions(graph);
        break;
      case 'circular':
        positions = this.calculateCircularPositions(graph);
        break;
      case 'hierarchical':
        positions = this.calculateHierarchicalPositions(graph);
        break;
      default:
        positions = await this.calculateForceDirectedPositions(graph);
    }

    // Apply clustering if requested
    let clusteredNodes = [...nodes];
    let clusters: any = undefined;
    if (clustering) {
      const clusteringResult = this.applyClustering(graph, layout);
      clusteredNodes = clusteringResult.nodes;
      clusters = clusteringResult.clusters;
    }

    return {
      nodes: clusteredNodes,
      edges: edges,
      positions,
      clusters
    };
  }

  /**
   * Calculate positions using force-directed algorithm (real implementation)
   */
  private async calculateForceDirectedPositions(graph: any): Promise<Record<string, { x: number; y: number }>> {
    const positions: Record<string, { x: number; y: number }> = {};
    const nodeIds = graph.nodes.map((n: any) => n.id);

    // Initialize random positions
    for (const nodeId of nodeIds) {
      positions[nodeId] = {
        x: (Math.random() - 0.5) * 400,
        y: (Math.random() - 0.5) * 400
      };
    }

    // Force-directed algorithm parameters
    const iterations = Math.min(100, Math.max(20, nodeIds.length)); // Adjust iterations based on graph size
    const k = 100; // Optimal distance
    const repulsionConstant = k * 0.5;
    const attractionConstant = k * 0.1;
    const maxDisplacement = 100;

    // Run force-directed algorithm
    for (let iter = 0; iter < iterations; iter++) {
      // Calculate repulsive forces
      for (let i = 0; i < nodeIds.length; i++) {
        for (let j = i + 1; j < nodeIds.length; j++) {
          const node1 = nodeIds[i];
          const node2 = nodeIds[j];

          const dx = positions[node1].x - positions[node2].x;
          const dy = positions[node1].y - positions[node2].y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 0.1; // Avoid division by zero

          // Repulsive force
          const repulsion = (repulsionConstant * repulsionConstant) / distance;
          const displacementX = (dx / distance) * repulsion;
          const displacementY = (dy / distance) * repulsion;

          positions[node1].x += displacementX;
          positions[node1].y += displacementY;
          positions[node2].x -= displacementX;
          positions[node2].y -= displacementY;
        }
      }

      // Calculate attractive forces
      for (const edge of graph.edges) {
        const sourcePos = positions[edge.source];
        const targetPos = positions[edge.target];

        if (sourcePos && targetPos) {
          const dx = sourcePos.x - targetPos.x;
          const dy = sourcePos.y - targetPos.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 0.1; // Avoid division by zero

          // Attractive force
          const attraction = (distance * distance) / k;
          const displacementX = (dx / distance) * attraction * attractionConstant;
          const displacementY = (dy / distance) * attraction * attractionConstant;

          sourcePos.x -= displacementX;
          sourcePos.y -= displacementY;
          targetPos.x += displacementX;
          targetPos.y += displacementY;
        }
      }

      // Limit displacement to prevent wild movements
      for (const nodeId of nodeIds) {
        const displacement = Math.sqrt(
          Math.pow(positions[nodeId].x, 2) +
          Math.pow(positions[nodeId].y, 2)
        );

        if (displacement > maxDisplacement) {
          positions[nodeId].x = (positions[nodeId].x / displacement) * maxDisplacement;
          positions[nodeId].y = (positions[nodeId].y / displacement) * maxDisplacement;
        }
      }
    }

    return positions;
  }

  /**
   * Calculate positions using circular layout
   */
  private calculateCircularPositions(graph: any): Record<string, { x: number; y: number }> {
    const positions: Record<string, { x: number; y: number }> = {};
    const nodeIds = graph.nodes.map((n: any) => n.id);

    // Order nodes by degree for better visual distribution
    const sortedNodeIds = [...nodeIds].sort((a, b) =>
      (graph.degrees.get(b) || 0) - (graph.degrees.get(a) || 0)
    );

    for (let i = 0; i < sortedNodeIds.length; i++) {
      const angle = (i / sortedNodeIds.length) * 2 * Math.PI;
      const radius = 150 + (graph.degrees.get(sortedNodeIds[i]) || 0) * 5; // Larger radius for higher-degree nodes
      positions[sortedNodeIds[i]] = {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius
      };
    }

    return positions;
  }

  /**
   * Calculate positions using hierarchical layout
   */
  private calculateHierarchicalPositions(graph: any): Record<string, { x: number; y: number }> {
    const positions: Record<string, { x: number; y: number }> = {};
    const nodeIds = graph.nodes.map((n: any) => n.id);

    // Find a root node (highest degree or first node)
    let rootId = nodeIds[0];
    let maxDegree = 0;
    for (const nodeId of nodeIds) {
      const degree = graph.degrees.get(nodeId) || 0;
      if (degree > maxDegree) {
        maxDegree = degree;
        rootId = nodeId;
      }
    }

    // BFS to assign levels
    const queue: string[] = [rootId];
    const visited = new Set<string>();
    const levels: Record<string, number> = {};
    levels[rootId] = 0;
    visited.add(rootId);

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = graph.adjacency.get(current) || [];

      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          levels[neighbor] = levels[current] + 1;
          queue.push(neighbor);
        }
      }
    }

    // Group nodes by level
    const levelGroups: Record<number, string[]> = {};
    for (const nodeId of nodeIds) {
      const level = levels[nodeId] !== undefined ? levels[nodeId] : 0;
      if (!levelGroups[level]) {
        levelGroups[level] = [];
      }
      levelGroups[level].push(nodeId);
    }

    // Position nodes by level
    const levelKeys = Object.keys(levelGroups).map(Number).sort((a, b) => a - b);
    for (let i = 0; i < levelKeys.length; i++) {
      const level = levelKeys[i];
      const nodesInLevel = levelGroups[level];
      const y = i * 100; // Vertical spacing by level

      for (let j = 0; j < nodesInLevel.length; j++) {
        const x = (j - (nodesInLevel.length - 1) / 2) * 80; // Horizontal spacing
        positions[nodesInLevel[j]] = { x, y };
      }
    }

    return positions;
  }

  /**
   * Apply clustering to nodes using community detection
   */
  private applyClustering(graph: any, layout: PyGraphistryLayout): { nodes: GraphNode[], clusters: any } {
    // Apply different clustering algorithms based on layout
    let clusters: any;

    switch (layout) {
      case 'force_directed':
        clusters = this.detectCommunitiesLouvain(graph);
        break;
      case 'circular':
        clusters = this.clusterByDegree(graph);
        break;
      case 'hierarchical':
        clusters = this.clusterByHierarchy(graph);
        break;
      default:
        clusters = this.detectCommunitiesLouvain(graph);
    }

    // Apply cluster information to nodes
    const clusteredNodes = graph.nodes.map((node: GraphNode) => {
      const newNode = { ...node };
      const clusterId = clusters.nodeClusters[node.id] || 0;
      (newNode as any).cluster = clusterId;
      (newNode as any).clusterLabel = `Cluster ${clusterId}`;
      return newNode;
    });

    return { nodes: clusteredNodes, clusters };
  }

  /**
   * Detect communities using Louvain algorithm (simplified version)
   */
  private detectCommunitiesLouvain(graph: any): { nodeClusters: Record<string, number>, clusters: any[] } {
    const nodeIds = graph.nodes.map((n: any) => n.id);
    const nodeClusters: Record<string, number> = {};
    const clusterMap: Map<string, number> = new Map();

    // Initialize each node as its own cluster
    for (let i = 0; i < nodeIds.length; i++) {
      clusterMap.set(nodeIds[i], i);
      nodeClusters[nodeIds[i]] = i;
    }

    // Simple modularity-based clustering
    let improved = true;
    let iteration = 0;
    const maxIterations = 10;

    while (improved && iteration < maxIterations) {
      improved = false;
      iteration++;

      for (const nodeId of nodeIds) {
        const neighbors = graph.adjacency.get(nodeId) || [];
        const currentCluster = clusterMap.get(nodeId)!;

        // Find the best neighboring cluster for this node
        const neighborClusters: Record<number, number> = {};
        for (const neighbor of neighbors) {
          const neighborCluster = clusterMap.get(neighbor)!;
          neighborClusters[neighborCluster] = (neighborClusters[neighborCluster] || 0) + 1;
        }

        // Find cluster with most connections
        let bestCluster = currentCluster;
        let maxConnections = 0;

        for (const [clusterId, connections] of Object.entries(neighborClusters)) {
          if (connections > maxConnections) {
            maxConnections = connections;
            bestCluster = parseInt(clusterId);
          }
        }

        // Update cluster if improvement found
        if (bestCluster !== currentCluster) {
          clusterMap.set(nodeId, bestCluster);
          nodeClusters[nodeId] = bestCluster;
          improved = true;
        }
      }
    }

    // Consolidate clusters (merge small clusters)
    const clusterCounts: Record<number, number> = {};
    for (const nodeId of nodeIds) {
      const clusterId = nodeClusters[nodeId];
      clusterCounts[clusterId] = (clusterCounts[clusterId] || 0) + 1;
    }

    // Create final clusters
    const clusters: any[] = [];
    const uniqueClusters = new Set(Object.values(nodeClusters));
    let clusterIndex = 0;

    for (const clusterId of uniqueClusters) {
      const members = nodeIds.filter(id => nodeClusters[id] === clusterId);
      clusters.push({
        id: clusterIndex++,
        size: members.length,
        members: members,
        density: members.length > 1 ? this.calculateClusterDensity(graph, members) : 0
      });
    }

    return { nodeClusters, clusters };
  }

  /**
   * Cluster by node degree
   */
  private clusterByDegree(graph: any): { nodeClusters: Record<string, number>, clusters: any[] } {
    const nodeIds = graph.nodes.map((n: any) => n.id);
    const nodeClusters: Record<string, number> = {};
    const clusters: any[] = [];

    // Calculate degree thresholds
    const degrees = Array.from(graph.degrees.values());
    if (degrees.length === 0) {
      // If no degrees calculated, assign all to cluster 0
      for (const nodeId of nodeIds) {
        nodeClusters[nodeId] = 0;
      }
      clusters.push({ id: 0, size: nodeIds.length, type: 'isolated' });
      return { nodeClusters, clusters };
    }

    const avgDegree = degrees.reduce((a, b) => a + b, 0) / degrees.length;
    const highDegreeThreshold = avgDegree * 1.5;

    for (const nodeId of nodeIds) {
      const degree = graph.degrees.get(nodeId) || 0;
      if (degree >= highDegreeThreshold) {
        nodeClusters[nodeId] = 0; // Hub cluster
      } else if (degree > 0) {
        nodeClusters[nodeId] = 1; // Connected cluster
      } else {
        nodeClusters[nodeId] = 2; // Isolated cluster
      }
    }

    clusters.push({ id: 0, size: Object.values(nodeClusters).filter(c => c === 0).length, type: 'hub' });
    clusters.push({ id: 1, size: Object.values(nodeClusters).filter(c => c === 1).length, type: 'connected' });
    clusters.push({ id: 2, size: Object.values(nodeClusters).filter(c => c === 2).length, type: 'isolated' });

    return { nodeClusters, clusters };
  }

  /**
   * Cluster by hierarchy levels
   */
  private clusterByHierarchy(graph: any): { nodeClusters: Record<string, number>, clusters: any[] } {
    const nodeIds = graph.nodes.map((n: any) => n.id);
    const nodeClusters: Record<string, number> = {};
    const clusters: any[] = [];

    // Use BFS from highest degree node to create level-based clusters
    let rootId = nodeIds[0];
    let maxDegree = 0;
    for (const nodeId of nodeIds) {
      const degree = graph.degrees.get(nodeId) || 0;
      if (degree > maxDegree) {
        maxDegree = degree;
        rootId = nodeId;
      }
    }

    // BFS to assign levels
    const queue: { node: string, level: number }[] = [{ node: rootId, level: 0 }];
    const visited = new Set<string>();
    const levels: Record<string, number> = {};
    levels[rootId] = 0;
    visited.add(rootId);

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = graph.adjacency.get(current.node) || [];

      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          levels[neighbor] = current.level + 1;
          queue.push({ node: neighbor, level: current.level + 1 });
        }
      }
    }

    // Assign clusters based on levels
    for (const nodeId of nodeIds) {
      const level = levels[nodeId] !== undefined ? levels[nodeId] : 0;
      nodeClusters[nodeId] = level;
    }

    // Create cluster info
    const uniqueLevels = new Set(Object.values(levels));
    for (const level of uniqueLevels) {
      const members = nodeIds.filter(id => levels[id] === level);
      clusters.push({ id: level, size: members.length, level: level, type: 'hierarchical' });
    }

    return { nodeClusters, clusters };
  }

  /**
   * Calculate cluster density
   */
  private calculateClusterDensity(graph: any, members: string[]): number {
    if (members.length < 2) return 0;

    let edgeCount = 0;
    for (const member of members) {
      const neighbors = graph.adjacency.get(member) || [];
      for (const neighbor of neighbors) {
        if (members.includes(neighbor)) {
          edgeCount++;
        }
      }
    }

    // Density = actual edges / possible edges
    const possibleEdges = (members.length * (members.length - 1)) / 2;
    return possibleEdges > 0 ? edgeCount / (2 * possibleEdges) : 0; // Divide by 2 because we counted each edge twice
  }

  /**
   * Generate local visualization based on processed graph
   */
  private async generateLocalVisualization(
    processedGraph: { nodes: any[], edges: any[], positions: Record<string, { x: number; y: number }> },
    layout: PyGraphistryLayout
  ): Promise<string> {
    // Create an SVG visualization of the graph
    const svgWidth = 800;
    const svgHeight = 600;

    // Find min/max coordinates to center the graph
    const positions = Object.values(processedGraph.positions);
    if (positions.length === 0) {
      return `data:text/html,<html><body><h2>Empty Graph Visualization</h2><p>No nodes to visualize</p></body></html>`;
    }

    const xs = positions.map(pos => pos.x);
    const ys = positions.map(pos => pos.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    // Scale factors to fit the SVG
    const scaleX = svgWidth / (maxX - minX + 100);
    const scaleY = svgHeight / (maxY - minY + 100);
    const scale = Math.min(scaleX, scaleY) * 0.8; // Leave some margin

    // Generate SVG content
    let svgContent = `<svg width="${svgWidth}" height="${svgHeight}" xmlns="http://www.w3.org/2000/svg">`;

    // Draw edges
    for (const edge of processedGraph.edges) {
      const sourcePos = processedGraph.positions[edge.source];
      const targetPos = processedGraph.positions[edge.target];

      if (sourcePos && targetPos) {
        const x1 = ((sourcePos.x - minX) * scale) + 50;
        const y1 = ((sourcePos.y - minY) * scale) + 50;
        const x2 = ((targetPos.x - minX) * scale) + 50;
        const y2 = ((targetPos.y - minY) * scale) + 50;

        svgContent += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#ccc" stroke-width="1"/>`;
      }
    }

    // Draw nodes
    for (const node of processedGraph.nodes) {
      const pos = processedGraph.positions[node.id];
      if (pos) {
        const x = ((pos.x - minX) * scale) + 50;
        const y = ((pos.y - minY) * scale) + 50;

        // Color based on cluster type if available
        let fill = '#4f46e5'; // Default blue
        if ((node as any).cluster === 'hub') {
          fill = '#ef4444'; // Red for hubs
        } else if ((node as any).cluster === 'peripheral') {
          fill = '#22c55e'; // Green for peripheral
        }

        svgContent += `<circle cx="${x}" cy="${y}" r="8" fill="${fill}" stroke="#fff" stroke-width="2"/>`;
        svgContent += `<text x="${x}" y="${y + 20}" text-anchor="middle" font-size="10" fill="#333">${node.label || node.id}</text>`;
      }
    }

    svgContent += '</svg>';

    // Create HTML page with the SVG
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>PyGraphistry Visualization - ${layout}</title>
        <style>
          body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
          .header { text-align: center; margin-bottom: 20px; }
          .stats { margin-bottom: 20px; text-align: center; }
          .graph-container { border: 1px solid #ddd; border-radius: 8px; overflow: auto; }
        </style>
      </head>
      <body>
        <div class="header">
          <h2>PyGraphistry Visualization</h2>
          <div class="stats">
            <p>Layout: ${layout} | Nodes: ${processedGraph.nodes.length} | Edges: ${processedGraph.edges.length}</p>
          </div>
        </div>
        <div class="graph-container">
          ${svgContent}
        </div>
      </body>
      </html>
    `;

    return `data:text/html;charset=utf-8,${encodeURIComponent(htmlContent)}`;
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    try {
      // Check for required nodes field
      const nodes = inputs.nodes;

      if (!nodes) {
        errors.push({
          field: 'nodes',
          message: 'Nodes array is required',
          severity: 'error'
        });
      } else if (!Array.isArray(nodes)) {
        errors.push({
          field: 'nodes',
          message: 'Nodes must be an array',
          severity: 'error'
        });
      } else if (nodes.length === 0) {
        errors.push({
          field: 'nodes',
          message: 'Nodes array cannot be empty',
          severity: 'error'
        });
      } else {
        // Validate individual nodes with additional safeguards
        for (let i = 0; i < nodes.length; i++) {
          if (i >= 10000) { // Prevent excessive validation
            console.warn(`Too many nodes (${nodes.length}), stopping validation at 10000`);
            break;
          }

          const node = nodes[i];
          if (!node || typeof node !== 'object') {
            errors.push({
              field: `nodes[${i}]`,
              message: `Node at index ${i} must be an object`,
              severity: 'error'
            });
          } else {
            // Validate node ID with additional checks
            if (!node.id) {
              errors.push({
                field: `nodes[${i}].id`,
                message: `Node at index ${i} must have an id`,
                severity: 'error'
              });
            } else if (typeof node.id !== 'string') {
              errors.push({
                field: `nodes[${i}].id`,
                message: `Node at index ${i} id must be a string`,
                severity: 'error'
              });
            } else if (node.id.length > 1000) {
              errors.push({
                field: `nodes[${i}].id`,
                message: `Node at index ${i} id is too long (max 1000 characters)`,
                severity: 'error'
              });
            }

            // Check for potential injection risks in node properties
            for (const [key, value] of Object.entries(node)) {
              if (typeof value === 'string' && (value.includes('<script') || value.includes('javascript:'))) {
                errors.push({
                  field: `nodes[${i}].${key}`,
                  message: `Node at index ${i} contains potential script injection in property ${key}`,
                  severity: 'error'
                });
              }
            }
          }
        }
      }

      // Check for required edges field
      const edges = inputs.edges;

      if (!edges) {
        errors.push({
          field: 'edges',
          message: 'Edges array is required',
          severity: 'error'
        });
      } else if (!Array.isArray(edges)) {
        errors.push({
          field: 'edges',
          message: 'Edges must be an array',
          severity: 'error'
        });
      } else if (edges.length > 50000) {
        errors.push({
          field: 'edges',
          message: 'Edges array is too large (max 50000 edges)',
          severity: 'error'
        });
      } else {
        // Validate individual edges with additional safeguards
        for (let i = 0; i < edges.length; i++) {
          if (i >= 50000) { // Prevent excessive validation
            console.warn(`Too many edges (${edges.length}), stopping validation at 50000`);
            break;
          }

          const edge = edges[i];
          if (!edge || typeof edge !== 'object') {
            errors.push({
              field: `edges[${i}]`,
              message: `Edge at index ${i} must be an object`,
              severity: 'error'
            });
          } else {
            if (!edge.source) {
              errors.push({
                field: `edges[${i}].source`,
                message: `Edge at index ${i} must have a source`,
                severity: 'error'
              });
            } else if (typeof edge.source !== 'string') {
              errors.push({
                field: `edges[${i}].source`,
                message: `Edge at index ${i} source must be a string`,
                severity: 'error'
              });
            }

            if (!edge.target) {
              errors.push({
                field: `edges[${i}].target`,
                message: `Edge at index ${i} must have a target`,
                severity: 'error'
              });
            } else if (typeof edge.target !== 'string') {
              errors.push({
                field: `edges[${i}].target`,
                message: `Edge at index ${i} target must be a string`,
                severity: 'error'
              });
            }

            // Check for potential injection risks in edge properties
            for (const [key, value] of Object.entries(edge)) {
              if (typeof value === 'string' && (value.includes('<script') || value.includes('javascript:'))) {
                errors.push({
                  field: `edges[${i}].${key}`,
                  message: `Edge at index ${i} contains potential script injection in property ${key}`,
                  severity: 'error'
                });
              }
            }
          }
        }
      }

      // Validate layout if provided
      if (inputs.layout && typeof inputs.layout === 'string') {
        const validLayouts: PyGraphistryLayout[] = ['force_directed', 'circular', 'hierarchical'];
        if (!validLayouts.includes(inputs.layout as PyGraphistryLayout)) {
          errors.push({
            field: 'layout',
            message: `Layout must be one of: ${validLayouts.join(', ')}`,
            severity: 'error'
          });
        }
      }

      // Validate clusteringMethod if provided
      if (inputs.clusteringMethod && typeof inputs.clusteringMethod === 'string') {
        const validMethods: PyGraphistryClusteringMethod[] = ['dbscan', 'kmeans'];
        if (!validMethods.includes(inputs.clusteringMethod as PyGraphistryClusteringMethod)) {
          errors.push({
            field: 'clusteringMethod',
            message: `Clustering method must be one of: ${validMethods.join(', ')}`,
            severity: 'error'
          });
        }
      }

      // Validate other inputs
      if (inputs.clustering !== undefined && typeof inputs.clustering !== 'boolean') {
        errors.push({
          field: 'clustering',
          message: 'Clustering must be a boolean',
          severity: 'error'
        });
      }

      if (inputs.enableGPUAcceleration !== undefined && typeof inputs.enableGPUAcceleration !== 'boolean') {
        errors.push({
          field: 'enableGPUAcceleration',
          message: 'enableGPUAcceleration must be a boolean',
          severity: 'error'
        });
      }

    } catch (validationError) {
      // If validation itself fails, add a generic error
      errors.push({
        field: 'inputs',
        message: `Validation error: ${validationError instanceof Error ? validationError.message : 'Unknown validation error'}`,
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    try {
      const schema: ParameterSchema = {
        type: 'object',
        properties: {
          layout: {
            type: 'string',
            description: 'Graph layout algorithm to use',
            enum: ['force_directed', 'circular', 'hierarchical'],
            default: 'force_directed'
          },
          clustering: {
            type: 'boolean',
            description: 'Enable clustering of nodes',
            default: false
          },
          clusteringMethod: {
            type: 'string',
            description: 'Clustering algorithm to use',
            enum: ['dbscan', 'kmeans'],
            default: 'dbscan'
          },
          enableGPUAcceleration: {
            type: 'boolean',
            description: 'Enable GPU acceleration for visualization',
            default: true
          },
          apiKey: {
            type: 'string',
            description: 'PyGraphistry API key',
            default: '',
            maxLength: 1000 // Prevent excessively long API keys
          },
          serverUrl: {
            type: 'string',
            description: 'PyGraphistry server URL',
            default: 'http://localhost:8000',
            pattern: '^https?://.+' // Basic URL validation
          },
          enableBackendExecution: {
            type: 'boolean',
            description: 'Use PyGraphistry service via integration library',
            default: true
          },
          backendUrl: {
            type: 'string',
            description: 'URL of the PyGraphistry service API',
            default: 'http://localhost:8000',
            pattern: '^https?://.+' // Basic URL validation
          }
        },
        required: []
      };

      // Validate the schema structure
      if (!schema || typeof schema !== 'object') {
        throw new Error('Generated schema is invalid');
      }

      return schema;
    } catch (schemaError) {
      errorLogger.logError(schemaError, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error generating PyGraphistry node parameter schema' } });
      // Return a minimal safe schema as fallback
      return {
        type: 'object',
        properties: {
          enableBackendExecution: {
            type: 'boolean',
            description: 'Use PyGraphistry service via integration library',
            default: true
          },
          backendUrl: {
            type: 'string',
            description: 'URL of the PyGraphistry service API',
            default: 'http://localhost:8000',
            pattern: '^https?://.+'
          }
        },
        required: []
      };
    }
  }

  /**
   * Cleanup when node is destroyed
   */
  destroy(): void {
    // No-op for now; backend connections are stateless HTTP calls.
    try {
      console.log(`[PyGraphistryNode] Node ${this.id} destroyed`);
    } catch (error) {
      console.error(`[PyGraphistryNode] Error during destruction of node ${this.id}:`, error);
      // Still consider the node destroyed even if cleanup fails
    }
  }

  private async postToBackend(endpoint: string, payload: Record<string, any>): Promise<PyGraphistryResult> {
    try {
      const backendUrl = (this.config.backendUrl as string | undefined) || '';

      if (!backendUrl) {
        return await apiClient.post<PyGraphistryResult>(endpoint, payload);
      }

      // Validate endpoint to prevent malicious URLs
      if (!endpoint || typeof endpoint !== 'string' || !endpoint.startsWith('/')) {
        throw new Error('Invalid endpoint: must be a relative path starting with "/"');
      }

      // Validate payload to prevent injection
      if (!payload || typeof payload !== 'object') {
        throw new Error('Invalid payload: must be an object');
      }

      let token;
      try {
        token = useAuthStore.getState().token;
      } catch (storeError) {
        console.warn('Auth store not available, proceeding without token:', storeError);
        token = null;
      }

      // Construct URL safely
      const url = new URL(endpoint, backendUrl).toString();

      // Validate that the constructed URL is valid
      try {
        new URL(url); // This will throw if the URL is invalid
      } catch (urlError) {
        throw new Error(`Invalid constructed URL: ${url}`);
      }

      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorDetails = response.statusText;
        try {
          const errorPayload = await response.json().catch(() => ({}));
          errorDetails = errorPayload?.error?.message || errorPayload?.message || response.statusText;
        } catch (parseError) {
          console.warn('Could not parse error response:', parseError);
        }

        throw new Error(`HTTP ${response.status}: ${errorDetails}`);
      }

      // Validate response before parsing
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Response is not JSON');
      }

      const result = await response.json();

      // Validate the result structure
      if (!result || typeof result !== 'object') {
        throw new Error('Invalid response format from server');
      }

      return result as PyGraphistryResult;
    } catch (error) {
      // Handle different types of errors
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(`Network error: Unable to reach the PyGraphistry server at ${this.config.backendUrl}. Please check your connection and server status.`);
      }

      if (error.name === 'AbortError') {
        throw new Error('Request timeout: The PyGraphistry server took too long to respond.');
      }

      if (error instanceof Error) {
        throw error; // Re-throw known errors
      }

      throw new Error('Unknown error occurred while communicating with the PyGraphistry server');
    }
  }
}

export default PyGraphistryNode;