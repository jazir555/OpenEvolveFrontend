import { default as React } from 'react';
import { NodeProps } from '@xyflow/react';
import { DecompositionNodeData } from '../../types';
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
export declare const DecompositionNodeComponent: React.MemoExoticComponent<(props: NodeProps<DecompositionNodeData>) => import("react/jsx-runtime").JSX.Element>;
export default DecompositionNodeComponent;
