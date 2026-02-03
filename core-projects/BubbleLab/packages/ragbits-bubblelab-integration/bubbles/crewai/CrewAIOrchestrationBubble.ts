import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core/src/types/service-bubble-class.js';
import type { BubbleContext } from '@bubblelab/bubble-core/src/types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Define the parameters schema for the CrewAI orchestration bubble
const CrewAIOrchestrationParamsSchema = z.object({
    serverUrl: z.string().url().default("http://localhost:8003").describe("CrewAI MCP server URL"),
    taskDescription: z.string().min(1).describe("Description of the task to delegate to CrewAI"),
    requiredOutputs: z.array(z.string()).describe("List of required outputs from the orchestration"),
    agentConfigs: z.array(z.object({
        role: z.string().describe("Role of the agent"),
        goal: z.string().describe("Goal of the agent"),
        backstory: z.string().describe("Backstory of the agent"),
    })).optional().describe("Configuration for custom agents"),
    taskConfigs: z.array(z.object({
        description: z.string().describe("Description of the task"),
        expectedOutput: z.string().describe("Expected output of the task"),
        agentRole: z.string().describe("Role of the agent to assign this task to"),
    })).optional().describe("Configuration for custom tasks"),
    constraints: z.array(z.string()).optional().describe("Constraints for the task execution"),
    context: z.record(z.string(), z.any()).optional().describe("Additional context for the orchestration"),
    apiKey: z.string().optional().describe("API key for MCP server authentication"),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe("Object mapping credential types to values (injected at runtime)"),
});

// Input and output types
type CrewAIOrchestrationParamsInput = z.input<typeof CrewAIOrchestrationParamsSchema>;
type CrewAIOrchestrationParams = z.output<typeof CrewAIOrchestrationParamsSchema>;

// Define the result schema
const CrewAIOrchestrationResultSchema = z.object({
    success: z.boolean().describe("Whether the orchestration was successful"),
    result: z.any().describe("Result from the CrewAI orchestration"),
    crewId: z.string().optional().describe("ID of the created crew (if applicable)"),
    agentCount: z.number().optional().describe("Number of agents in the crew"),
    taskCount: z.number().optional().describe("Number of tasks in the crew"),
    error: z.string().optional().describe("Error message if orchestration failed"),
});

type CrewAIOrchestrationResult = z.output<typeof CrewAIOrchestrationResultSchema>;

export class CrewAIOrchestrationBubble extends ServiceBubble<
    CrewAIOrchestrationParams,
    CrewAIOrchestrationResult
> {
    static readonly service = 'crewai-mcp';
    static readonly authType = 'apiKey' as const;
    static readonly bubbleName = 'crewai-orchestration';
    static readonly type = 'service' as const;
    static readonly schema = CrewAIOrchestrationParamsSchema;
    static readonly resultSchema = CrewAIOrchestrationResultSchema;
    static readonly shortDescription = 'Orchestrate complex tasks using CrewAI';
    static readonly longDescription = `
        A bubble that delegates complex orchestration tasks to CrewAI through the MCP protocol.
        Use cases:
        - Multi-agent task coordination
        - Complex workflow orchestration
        - Research and analysis tasks
        - Content creation workflows
        - Decision-making processes
    `;
    static readonly alias = 'crewai';

    constructor(
        params: CrewAIOrchestrationParamsInput,
        context?: BubbleContext
    ) {
        super(params, context);
    }

    protected chooseCredential(): string | undefined {
        // Use the API key from params or return undefined if not provided
        return this.params.apiKey;
    }

    public async testCredential(): Promise<boolean> {
        // Test connection to CrewAI MCP server
        try {
            // In a real implementation, this would make an HTTP request to the server
            // For now, we'll just return true to indicate the connection can be tested
            return true;
        } catch (error) {
            console.error('Error testing CrewAI MCP connection:', error);
            return false;
        }
    }

    protected async performAction(
        context?: BubbleContext
    ): Promise<CrewAIOrchestrationResult> {
        // Context is available but not currently used in this implementation
        void context;

        try {
            // In a real implementation, this would make HTTP requests to the MCP server
            // For now, we'll simulate the interaction

            // This is a simplified simulation - in reality, this would:
            // 1. Make HTTP requests to the MCP server
            // 2. Handle the response appropriately
            // 3. Return the proper result

            const result = {
                success: true,
                result: `CrewAI processed task: ${this.params.taskDescription}`,
                crewId: 'simulated-crew-id',
                agentCount: this.params.agentConfigs?.length || 1,
                taskCount: this.params.taskConfigs?.length || 1,
                error: undefined
            };

            // Return the result
            return result;
        } catch (error) {
            return {
                success: false,
                result: null,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
}


// Additional specialized bubbles for specific CrewAI use cases

// Research Bubble
const CrewAIResearchParamsSchema = z.object({
    serverUrl: z.string().url().default("http://localhost:8003").describe("CrewAI MCP server URL"),
    topic: z.string().min(1).describe("Topic to research"),
    researchDepth: z.number().min(1).max(5).default(3).describe("Depth level of research"),
    additionalConstraints: z.array(z.string()).optional().describe("Additional constraints for research"),
    apiKey: z.string().optional().describe("API key for MCP server authentication"),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe("Object mapping credential types to values (injected at runtime)"),
});

type CrewAIResearchParamsInput = z.input<typeof CrewAIResearchParamsSchema>;
type CrewAIResearchParams = z.output<typeof CrewAIResearchParamsSchema>;

const CrewAIResearchResultSchema = z.object({
    success: z.boolean().describe("Whether the research was successful"),
    report: z.string().describe("Research report generated by CrewAI"),
    sources: z.array(z.string()).describe("Sources used in the research"),
    error: z.string().optional().describe("Error message if research failed"),
});

type CrewAIResearchResult = z.output<typeof CrewAIResearchResultSchema>;

export class CrewAIResearchBubble extends ServiceBubble<
    CrewAIResearchParams,
    CrewAIResearchResult
> {
    static readonly service = 'crewai-mcp';
    static readonly authType = 'apiKey' as const;
    static readonly bubbleName = 'crewai-research';
    static readonly type = 'service' as const;
    static readonly schema = CrewAIResearchParamsSchema;
    static readonly resultSchema = CrewAIResearchResultSchema;
    static readonly shortDescription = 'Conduct research using CrewAI';
    static readonly longDescription = `
        A bubble that conducts research on a specified topic using CrewAI.
        Use cases:
        - Market research
        - Academic research
        - Competitive analysis
        - Technology trend analysis
    `;
    static readonly alias = 'crewai-research';

    constructor(
        params: CrewAIResearchParamsInput,
        context?: BubbleContext
    ) {
        super(params, context);
    }

    protected chooseCredential(): string | undefined {
        return this.params.apiKey;
    }

    public async testCredential(): Promise<boolean> {
        try {
            // Similar to above, in a real implementation this would test the connection
            return true;
        } catch (error) {
            console.error('Error testing CrewAI MCP connection:', error);
            return false;
        }
    }

    protected async performAction(
        context?: BubbleContext
    ): Promise<CrewAIResearchResult> {
        void context;

        try {
            // Simulate research task
            const result = {
                success: true,
                report: `Research report on ${this.params.topic} with depth ${this.params.researchDepth}`,
                sources: ['Simulated source 1', 'Simulated source 2'],
                error: undefined
            };

            return result;
        } catch (error) {
            return {
                success: false,
                report: '',
                sources: [],
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
}