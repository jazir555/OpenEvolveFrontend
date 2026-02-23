/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Memory Integration Example
 *
 * Demonstrates how to use ICR's Contextual Mode with Graphiti memory integration.
 * This example shows:
 * 1. Setting up the memory agent
 * 2. Making memory-enhanced contextual requests
 * 3. Retrieving and storing memories
 * 4. Learning from session outcomes
 */
import { ICRAdapter } from '../src/adapter';
import { EnhancedICRMemoryAgent } from '../src/memory/memory-agent';
declare function setupMemoryAgent(): Promise<ICRAdapter>;
declare function example1_MemoryEnhancedRequest(icrAdapter: ICRAdapter): Promise<void>;
declare function example2_DirectMemoryUsage(memoryAgent: EnhancedICRMemoryAgent): Promise<void>;
declare function example3_ManualMemoryStorage(memoryAgent: EnhancedICRMemoryAgent, sessionId: string): Promise<void>;
declare function example4_SessionLearning(memoryAgent: EnhancedICRMemoryAgent, sessionId: string): Promise<void>;
declare function example5_PatternAnalysis(memoryAgent: EnhancedICRMemoryAgent): Promise<void>;
export { setupMemoryAgent, example1_MemoryEnhancedRequest, example2_DirectMemoryUsage, example3_ManualMemoryStorage, example4_SessionLearning, example5_PatternAnalysis };
//# sourceMappingURL=memory-usage.d.ts.map