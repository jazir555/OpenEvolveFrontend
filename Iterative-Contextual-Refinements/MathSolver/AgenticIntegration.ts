/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Agentic Mode Integration for MathSolver
 * 
 * This file provides integration helpers for adding MathSolver tools
to the Agentic mode. Import and use these functions to extend Agentic
 * with mathematical reasoning capabilities.
 * 
 * API Version: 1.1.0 (matches backend)
 */

import { 
    ToolCall, 
    AgenticState, 
    AgenticMessage,
    parseAgentResponseWithSegments 
} from '../Agentic/AgenticCore';
import {
    executeMathToolCall,
    isMathTool,
    MATH_TOOLS_PROMPT,
    type MathToolCall
} from './MathTools';

// Re-export for convenience
export { MATH_TOOLS_PROMPT, isMathTool };
export type { MathToolCall };

/**
 * Extended ToolCall type including math tools
 */
export type ExtendedToolCall = ToolCall | MathToolCall;

/**
 * Extended system prompt with math tools
 */
export function getExtendedSystemPrompt(basePrompt: string): string {
    return `${basePrompt}\n\n${MATH_TOOLS_PROMPT}`;
}

/**
 * Execute any tool call (standard or math)
 */
export async function executeExtendedToolCall(
    content: string,
    toolCall: ExtendedToolCall,
    modelName?: string,
    agenticPromptsManager?: any,
    sessionId?: string
): Promise<string> {
    // Check if it's a math tool
    if (isMathTool((toolCall as any).type)) {
        return await executeMathToolCall(toolCall as MathToolCall);
    }
    
    // Otherwise, it's a standard tool - import and call from AgenticCore
    const { executeToolCall } = await import('../Agentic/AgenticCore');
    return await executeToolCall(content, toolCall as ToolCall, modelName, agenticPromptsManager, sessionId);
}

/**
 * Parse agent response including math tool calls
 * 
 * This extends the standard parseAgentResponseWithSegments to recognize
 * math tool syntax in agent responses.
 */
export function parseExtendedResponse(response: string): {
    actions: ExtendedToolCall[];
    segments: any[];
} {
    const parsed = parseAgentResponseWithSegments(response);
    
    // Math tools use the same bracket syntax as other tools
    // They are already parsed by parseAgentResponseWithSegments
    // We just need to type them correctly
    
    const extendedActions: ExtendedToolCall[] = parsed.actions.map(action => {
        if ('type' in action && isMathTool(action.type)) {
            return action as MathToolCall;
        }
        return action as ToolCall;
    });
    
    return {
        actions: extendedActions,
        segments: parsed.segments
    };
}

/**
 * Math-enabled conversation manager
 * 
 * Wraps the standard AgenticConversationManager with math tool support.
 */
export class MathEnabledConversationManager {
    private baseManager: any; // AgenticConversationManager
    private mathEnabled: boolean;
    
    constructor(
        originalContent: string,
        systemPrompt: string,
        verifierPrompt?: string,
        enableMath: boolean = true
    ) {
        // Import and create base manager
        const { AgenticConversationManager } = require('../Agentic/AgenticCoreLangchain');
        
        const extendedPrompt = enableMath ? getExtendedSystemPrompt(systemPrompt) : systemPrompt;
        this.baseManager = new AgenticConversationManager(originalContent, extendedPrompt, verifierPrompt);
        this.mathEnabled = enableMath;
    }
    
    // Delegate methods to base manager
    async addAgentMessage(content: string): Promise<void> {
        return this.baseManager.addAgentMessage(content);
    }
    
    async addSystemMessage(content: string): Promise<void> {
        return this.baseManager.addSystemMessage(content);
    }
    
    async addUserMessage(content: string): Promise<void> {
        return this.baseManager.addUserMessage(content);
    }
    
    async getConversationHistory(): Promise<string> {
        return this.baseManager.getConversationHistory();
    }
    
    async getStructuredMessages(): Promise<Array<{ role: 'system' | 'assistant' | 'user'; content: string }>> {
        return this.baseManager.getStructuredMessages();
    }
    
    async buildPrompt(): Promise<string> {
        return this.baseManager.buildPrompt();
    }
    
    async buildStructuredPrompt(): Promise<Array<{ role: 'system' | 'assistant' | 'user'; content: string }>> {
        return this.baseManager.buildStructuredPrompt();
    }
    
    updateCurrentContent(newContent: string): void {
        this.baseManager.updateCurrentContent(newContent);
    }
    
    getCurrentContent(): string {
        return this.baseManager.getCurrentContent();
    }
    
    getSystemPrompt(): string {
        return this.baseManager.getSystemPrompt();
    }
    
    // Math-specific methods
    isMathEnabled(): boolean {
        return this.mathEnabled;
    }
    
    enableMath(): void {
        this.mathEnabled = true;
    }
    
    disableMath(): void {
        this.mathEnabled = false;
    }
}

/**
 * Integration status checker
 */
export async function checkMathSolverIntegration(): Promise<{
    available: boolean;
    backendConnected: boolean;
    toolsLoaded: boolean;
    version?: string;
    details: Record<string, any>;
}> {
    const details: Record<string, any> = {};
    
    try {
        // Check if we can import math solver
        const { MathSolverCore, MATH_SOLVER_VERSION } = await import('./MathSolverCore');
        const core = new MathSolverCore();
        details.mathSolverCoreLoaded = true;
        details.version = MATH_SOLVER_VERSION;
        
        // Check backend health
        const health = await core.checkBackendHealth();
        details.backendHealth = health;
        
        // Get API info if available
        try {
            const apiInfo = await core['api'].getApiInfo();
            details.apiVersion = apiInfo.version;
            details.apiEndpoints = apiInfo.endpoints;
        } catch (e) {
            details.apiInfoError = 'Could not fetch API info';
        }
        
        return {
            available: true,
            backendConnected: health.available,
            toolsLoaded: true,
            version: MATH_SOLVER_VERSION,
            details
        };
    } catch (error) {
        details.error = error instanceof Error ? error.message : 'Unknown error';
        return {
            available: false,
            backendConnected: false,
            toolsLoaded: false,
            details
        };
    }
}

// Default export with all integration utilities
export default {
    getExtendedSystemPrompt,
    executeExtendedToolCall,
    parseExtendedResponse,
    isMathTool,
    MathEnabledConversationManager,
    checkMathSolverIntegration,
    MATH_TOOLS_PROMPT
};
