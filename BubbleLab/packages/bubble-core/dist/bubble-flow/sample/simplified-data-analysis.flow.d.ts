import { BubbleFlow } from '../bubble-flow-class.js';
import type { BubbleFlowOperationResult } from '../../types/bubble.js';
import type { BubbleTriggerEventRegistry } from '@bubblelab/shared-schemas';
/**
 * SimplifiedDataAnalysisFlow - Example showing the power of WorkflowBubbles
 *
 * This example demonstrates how WorkflowBubbles dramatically simplify BubbleFlow creation.
 *
 * BEFORE WorkflowBubbles (original integration test):
 * - 200+ lines of code
 * - Manual credential management
 * - Complex error handling
 * - Multiple service bubble orchestration
 * - Manual Slack channel discovery
 * - Manual AI formatting prompts
 *
 * AFTER WorkflowBubbles (this example):
 * - ~50 lines of clean, readable code
 * - Automatic credential injection
 * - Built-in error handling
 * - High-level workflow abstractions
 * - User-friendly parameter names
 * - TypeScript intellisense support
 */
export declare class SimplifiedDataAnalysisFlow extends BubbleFlow<'webhook/http'> {
    constructor();
    handle(payload: BubbleTriggerEventRegistry['webhook/http']): Promise<BubbleFlowOperationResult>;
}
//# sourceMappingURL=simplified-data-analysis.flow.d.ts.map