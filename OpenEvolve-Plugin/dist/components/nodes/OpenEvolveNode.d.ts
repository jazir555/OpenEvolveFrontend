import { default as React } from 'react';
import { NodeProps } from '@xyflow/react';
import { OpenEvolveNodeData } from '../types/nodeTypes';
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
export declare const OpenEvolveNode: React.MemoExoticComponent<(props: NodeProps<OpenEvolveNodeData>) => import("react/jsx-runtime").JSX.Element>;
export default OpenEvolveNode;
