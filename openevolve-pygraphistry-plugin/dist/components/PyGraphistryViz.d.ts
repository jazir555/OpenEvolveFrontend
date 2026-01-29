import { default as React } from 'react';
import { GraphNode, GraphEdge } from '../types/plugin-types';

interface Props {
    nodes: GraphNode[];
    edges: GraphEdge[];
    height?: string | number;
    autoGenerate?: boolean;
}
export declare const PyGraphistryViz: React.FC<Props>;
export {};
