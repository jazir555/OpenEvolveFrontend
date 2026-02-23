/**
 * ROMA Execution Panel Component
 *
 * A comprehensive panel for monitoring and managing ROMA task executions.
 * Displays execution status, results, statistics, and provides controls.
 */
import React from 'react';
/**
 * RomaExecutionPanel Props
 */
export interface RomaExecutionPanelProps {
    executionId?: string;
    onClose?: () => void;
    showFullHistory?: boolean;
}
/**
 * RomaExecutionPanel Component
 *
 * Provides a full-featured execution panel with:
 * - Task input form
 * - Execution status display
 * - Real-time progress monitoring
 * - Execution history
 * - Statistics dashboard
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <RomaExecutionPanel
 *       showFullHistory={true}
 *       onClose={() => console.log('Panel closed')}
 *     />
 *   );
 * }
 * ```
 */
export declare const RomaExecutionPanel: React.FC<RomaExecutionPanelProps>;
export default RomaExecutionPanel;
//# sourceMappingURL=RomaExecutionPanel.d.ts.map