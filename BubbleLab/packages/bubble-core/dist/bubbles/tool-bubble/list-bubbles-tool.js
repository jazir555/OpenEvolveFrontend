import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { BubbleFactory } from '../../bubble-factory.js';
// Define the parameters schema
const ListBubblesToolParamsSchema = z.object({});
// Result schema for validation
const ListBubblesToolResultSchema = z.object({
    bubbles: z
        .array(z.object({
        name: z.string().describe('Name of the bubble'),
        alias: z.string().optional().describe('Short alias for the bubble'),
        shortDescription: z
            .string()
            .describe('Brief description of the bubble functionality'),
        useCase: z.string().describe('Primary use cases for the bubble'),
        type: z.string().describe('Type of bubble (service, workflow, tool)'),
    }))
        .describe('Array of bubble information objects'),
    totalCount: z.number().describe('Total number of bubbles in the registry'),
    success: z.boolean().describe('Whether the operation was successful'),
    error: z.string().describe('Error message if operation failed'),
});
export class ListBubblesTool extends ToolBubble {
    // Required static metadata
    static bubbleName = 'list-bubbles-tool';
    static schema = ListBubblesToolParamsSchema;
    static resultSchema = ListBubblesToolResultSchema;
    static shortDescription = 'Lists all available bubbles in the registry';
    static longDescription = `
    A tool bubble that provides a comprehensive list of all registered bubbles in the NodeX system.
    
    Returns information about each bubble including:
    - Bubble name and alias
    - Short description
    - Extracted use cases
    - Bubble type (service, workflow, tool)
    
    Use cases:
    - AI agent discovery of available capabilities
    - System introspection and documentation
    - Dynamic tool selection for workflow building
  `;
    static alias = 'list';
    static type = 'tool';
    constructor(params = {}, context) {
        super(params, context);
    }
    async performAction(context) {
        void context; // Context available but not currently used
        const factory = new BubbleFactory();
        await factory.registerDefaults();
        const allMetadata = factory.getAllMetadata();
        // Filter out any metadata that is undefined
        const filteredMetadata = allMetadata.filter((metadata) => metadata !== undefined);
        const bubbles = filteredMetadata.map((metadata) => ({
            name: metadata.name,
            alias: metadata.alias,
            shortDescription: metadata.shortDescription,
            useCase: this.extractUseCaseFromDescription(metadata.longDescription),
            type: metadata.type,
        }));
        return {
            bubbles,
            totalCount: bubbles.length,
            success: true,
            error: '',
        };
    }
    extractUseCaseFromDescription(longDescription) {
        // Extract use cases from long description
        const useCaseMatch = longDescription.match(/Use cases?:\s*\n?(.*?)(?:\n\n|\n\s*-|\n\s*\*|$)/s);
        if (useCaseMatch) {
            return useCaseMatch[1]
                .trim()
                .replace(/\n\s*-\s*/g, ', ')
                .replace(/\n/g, ' ');
        }
        // Fallback to short description if no use cases found
        return 'General purpose bubble for various workflow needs';
    }
}
//# sourceMappingURL=list-bubbles-tool.js.map